"""
=========================================================
KITTI Odometry Replay Engine
Confidence-Driven Predictive Localization Dashboard
=========================================================

Loads KITTI Odometry sequences (colour images + ground-truth
SE(3) poses) and emits the same ``packet`` dict schema as
``KITTIReplay`` so the rest of the dashboard pipeline works
without any modifications.

Key differences from KITTI Raw:
  - No OXTS / GPS / IMU hardware data → values are synthesised
    from the ground-truth pose matrices.
  - Sequences 00–10 have ground-truth poses; 11–21 are test
    sequences with no poses (XYZ will be zero).

Author: Research Project
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from config import (
    KITTI_ODOMETRY_CALIB,
    KITTI_ODOMETRY_POSES,
    KITTI_ODOMETRY_SEQUENCES,
    KITTI_ODOMETRY_POSE_SEQUENCES,
)


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

# Reference origin used to convert metric XZ displacements → lat/lon offsets.
# Using a central-Germany reference consistent with KITTI recordings.
_REF_LAT = 48.9840101  # degrees
_REF_LON = 8.4357023   # degrees
_METERS_PER_DEG_LAT = 111_320.0


def _xyz_to_latlon(x: float, z: float) -> tuple[float, float]:
    """Convert camera-frame X (right) / Z (forward) offsets in metres to lat/lon."""
    m_per_deg_lon = _METERS_PER_DEG_LAT * math.cos(math.radians(_REF_LAT))
    lat = _REF_LAT + z / _METERS_PER_DEG_LAT
    lon = _REF_LON + x / m_per_deg_lon
    return lat, lon


def _rotation_to_euler(R: np.ndarray) -> tuple[float, float, float]:
    """Extract roll / pitch / yaw (in radians) from a 3×3 rotation matrix."""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    return roll, pitch, yaw


# ---------------------------------------------------------------------------
# Pose loader
# ---------------------------------------------------------------------------

def _load_poses(pose_file: Path) -> list[np.ndarray]:
    """Load a KITTI pose file and return a list of 3×4 SE(3) matrices."""
    poses: list[np.ndarray] = []
    if not pose_file.exists():
        return poses
    with pose_file.open() as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            if len(values) == 12:
                poses.append(np.array(values, dtype=np.float64).reshape(3, 4))
    return poses


# ---------------------------------------------------------------------------
# Image list loader
# ---------------------------------------------------------------------------

def _collect_images(sequence_dir: Path) -> list[Path]:
    """Collect sorted PNG paths from image_2 (left colour camera)."""
    img_dir = sequence_dir / "image_2"
    if not img_dir.exists():
        return []
    return sorted(img_dir.glob("*.png"))


# ---------------------------------------------------------------------------
# Timestamps loader
# ---------------------------------------------------------------------------

def _load_timestamps(sequence_dir: Path) -> list[float]:
    """Load timestamps (in seconds) from times.txt."""
    ts_file = sequence_dir / "times.txt"
    if not ts_file.exists():
        return []
    with ts_file.open() as f:
        return [float(line.strip()) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Main replay class
# ---------------------------------------------------------------------------

class KITTIOdometryReplay:
    """
    Replay Engine for KITTI Odometry Dataset.

    Outputs the same ``packet`` dict schema as ``KITTIReplay`` so that the
    existing ``FusionEngine``, charts panels, and telemetry panels all work
    without modification.

    Packet fields produced
    ----------------------
    - frame_id, timestamp, image
    - latitude, longitude, altitude  (synthesised from SE(3) pose)
    - roll, pitch, yaw               (derived from rotation sub-matrix)
    - vf, vl, vu                     (finite-difference velocity)
    - ax, ay, az                     (finite-difference acceleration)
    - wx, wy, wz                     (angular velocity from Δ yaw/pitch/roll)
    - numsats, pos_accuracy, vel_accuracy, navstat, posmode, velmode, orimode
      (synthetic constants — no real OXTS in Odometry)
    - oxts, gps, imu                 (stub dicts for compatibility)
    """

    def __init__(self, sequence: str = "00") -> None:
        self.sequence = sequence
        self._seq_dir = KITTI_ODOMETRY_SEQUENCES / sequence
        self._pose_file = KITTI_ODOMETRY_POSES / f"{sequence}.txt"

        self._images: list[Path] = []
        self._poses: list[np.ndarray] = []
        self._timestamps: list[float] = []

        self.total_frames: int = 0
        self.current_frame: int = 0

        # Cache synthesised derivatives between frames for velocity / accel
        self._prev_xyz: np.ndarray | None = None
        self._prev_vel: np.ndarray | None = None
        self._prev_euler: np.ndarray | None = None
        self._prev_dt: float = 0.1

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> None:
        print("=" * 60)
        print("Loading KITTI Odometry Dataset")
        print("=" * 60)

        self._images = _collect_images(self._seq_dir)
        self._poses = _load_poses(self._pose_file)
        self._timestamps = _load_timestamps(self._seq_dir)

        n_images = len(self._images)
        n_poses = len(self._poses)

        # Sequences 11–21 have no poses — synthesise zero poses
        if n_poses == 0 and n_images > 0:
            zero = np.eye(3, 4, dtype=np.float64)
            self._poses = [zero.copy() for _ in range(n_images)]

        # Align lengths
        self.total_frames = min(n_images, len(self._poses))
        if not self._timestamps or len(self._timestamps) < self.total_frames:
            self._timestamps = [
                i * 0.1 for i in range(self.total_frames)
            ]

        print(f"Sequence : {self.sequence}")
        print(f"Frames   : {self.total_frames}")
        print(f"Has GT   : {'Yes' if self.sequence in KITTI_ODOMETRY_POSE_SEQUENCES else 'No (test seq)'}")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.current_frame = 0
        self._prev_xyz = None
        self._prev_vel = None
        self._prev_euler = None

    def has_next(self) -> bool:
        return self.current_frame < self.total_frames

    def current_index(self) -> int:
        return self.current_frame

    def __len__(self) -> int:
        return self.total_frames

    def seek(self, frame_index: int) -> None:
        frame_index = max(0, min(frame_index, self.total_frames - 1))
        if frame_index < self.current_frame:
            # Seeking backwards — reset caches
            self._prev_xyz = None
            self._prev_vel = None
            self._prev_euler = None
        self.current_frame = frame_index

    def progress(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return self.current_frame / self.total_frames

    # ------------------------------------------------------------------
    # Frame reading
    # ------------------------------------------------------------------

    def next_frame(self) -> dict[str, Any] | None:
        if not self.has_next():
            return None
        packet = self.get_frame(self.current_frame)
        self.current_frame += 1
        return packet

    def get_frame(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= self.total_frames:
            raise IndexError(f"Frame index {index} out of range [0, {self.total_frames}).")

        # ---- Image -------------------------------------------------------
        image = Image.open(self._images[index]).convert("RGB")

        # ---- Pose --------------------------------------------------------
        pose = self._poses[index]          # 3×4
        R = pose[:, :3]                    # 3×3 rotation
        t = pose[:, 3]                     # (3,) translation in camera frame

        # Camera-frame: X=right, Y=down, Z=forward
        x, y, z = float(t[0]), float(t[1]), float(t[2])
        roll, pitch, yaw = _rotation_to_euler(R)
        lat, lon = _xyz_to_latlon(x, z)
        alt = -y  # camera Y is downward; altitude increases upward

        # ---- Timestamp ---------------------------------------------------
        ts = self._timestamps[index]

        # ---- Velocity (finite difference) --------------------------------
        dt = self._prev_dt
        if index > 0 and len(self._timestamps) > index:
            dt = max(self._timestamps[index] - self._timestamps[index - 1], 1e-6)

        curr_xyz = np.array([x, y, z])
        curr_euler = np.array([roll, pitch, yaw])

        if self._prev_xyz is not None:
            vel = (curr_xyz - self._prev_xyz) / dt
        else:
            vel = np.zeros(3)

        vf = float(vel[2])   # forward  (Z)
        vl = float(vel[0])   # lateral  (X)
        vu = float(-vel[1])  # vertical (camera Y is down → flip)

        # ---- Acceleration ------------------------------------------------
        if self._prev_vel is not None:
            acc = (vel - self._prev_vel) / dt
        else:
            acc = np.zeros(3)

        ax, ay, az = float(acc[0]), float(acc[1]), float(acc[2])

        # ---- Angular velocity -------------------------------------------
        if self._prev_euler is not None:
            deuler = (curr_euler - self._prev_euler) / dt
        else:
            deuler = np.zeros(3)

        wx, wy, wz = float(deuler[0]), float(deuler[1]), float(deuler[2])

        # ---- Update caches -----------------------------------------------
        self._prev_xyz = curr_xyz
        self._prev_vel = vel
        self._prev_euler = curr_euler
        self._prev_dt = dt

        # ---- Synthetic GPS quality fields --------------------------------
        # KITTI Odometry has no real OXTS; provide plausible constants
        # so the downstream sensor-quality scoring doesn't crash.
        stub_gps = _OxtpStub(
            lat=lat, lon=lon, alt=alt,
            roll=roll, pitch=pitch, yaw=yaw,
            vf=vf, vl=vl, vu=vu,
            ax=ax, ay=ay, az=az,
            wx=wx, wy=wy, wz=wz,
            numsats=10, pos_accuracy=0.05, vel_accuracy=0.03,
            navstat=4, posmode=4, velmode=4, orimode=4,
        )

        return {
            # ------------------------------------------
            # Metadata
            # ------------------------------------------
            "frame_id": index,
            "timestamp": ts,
            "image": image,

            # ------------------------------------------
            # GPS (synthesised from pose)
            # ------------------------------------------
            "latitude": lat,
            "longitude": lon,
            "altitude": alt,

            # ------------------------------------------
            # Vehicle orientation
            # ------------------------------------------
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,

            # ------------------------------------------
            # Velocity (m/s)
            # ------------------------------------------
            "vf": vf,
            "vl": vl,
            "vu": vu,

            # ------------------------------------------
            # Acceleration (m/s²)
            # ------------------------------------------
            "ax": ax,
            "ay": ay,
            "az": az,

            # ------------------------------------------
            # Angular velocity (rad/s)
            # ------------------------------------------
            "wx": wx,
            "wy": wy,
            "wz": wz,

            # ------------------------------------------
            # GPS quality (synthetic)
            # ------------------------------------------
            "numsats": 10,
            "pos_accuracy": 0.05,
            "vel_accuracy": 0.03,
            "navstat": 4,
            "posmode": 4,
            "velmode": 4,
            "orimode": 4,

            # ------------------------------------------
            # Raw stubs (for compatibility)
            # ------------------------------------------
            "oxts": stub_gps,
            "gps": stub_gps,
            "imu": stub_gps,
        }

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def info(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "frames": self.total_frames,
            "current_frame": self.current_frame,
            "progress": round(self.progress() * 100, 2),
        }


# ---------------------------------------------------------------------------
# OXTS-compatible stub so code that accesses .packet attributes doesn't crash
# ---------------------------------------------------------------------------

class _OxtpStub:
    """Minimal stub mimicking a pykitti OxtsPacket namedtuple."""

    __slots__ = (
        "lat", "lon", "alt",
        "roll", "pitch", "yaw",
        "vf", "vl", "vu",
        "ax", "ay", "az",
        "wx", "wy", "wz",
        "numsats", "pos_accuracy", "vel_accuracy",
        "navstat", "posmode", "velmode", "orimode",
        "packet",
    )

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)
        # Some downstream code does oxts.packet.X
        self.packet = self
