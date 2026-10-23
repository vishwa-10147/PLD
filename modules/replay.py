"""
=========================================================
KITTI Replay Engine
Confidence-Driven Predictive Localization Dashboard
=========================================================

Loads synchronized KITTI camera and OXTS data
using the pykitti library.

Author: Research Project
"""

import pykitti

from config import *


class KITTIReplay:
    """
    Replay Engine for KITTI Raw Dataset.

    Each replay frame contains:

    - Camera Image
    - GPS
    - IMU
    - Timestamp
    - Vehicle Motion
    - Sensor Quality Information
    """

    def __init__(self, drive_name=DEFAULT_DRIVE):

        self.drive_name = drive_name

        self.date = KITTI_DATE

        # Example:
        # 2011_09_26_drive_0022_sync
        # --------------------^^^^
        self.drive = drive_name.split("_")[-2]

        self.dataset = None

        self.total_frames = 0

        self.current_frame = 0

    # ======================================================
    # Load Dataset
    # ======================================================

    def load(self):

        print("=" * 60)
        print("Loading KITTI Dataset")
        print("=" * 60)

        self.dataset = pykitti.raw(
            str(KITTI_ROOT),
            self.date,
            self.drive
        )

        self.total_frames = len(self.dataset.timestamps)

        print(f"Drive  : {self.drive_name}")
        print(f"Frames : {self.total_frames}")

        print("=" * 60)

    # ======================================================
    # Reset Replay
    # ======================================================

    def reset(self):

        self.current_frame = 0

    # ======================================================
    # Replay Status
    # ======================================================

    def has_next(self):

        return self.current_frame < self.total_frames

    # ======================================================
    # Current Position
    # ======================================================

    def current_index(self):

        return self.current_frame

    # ======================================================
    # Total Frames
    # ======================================================

    def __len__(self):

        return self.total_frames

    # ======================================================
    # Read One Frame
    # ======================================================

    def next_frame(self):

        if not self.has_next():
            return None

        packet = self.get_frame(self.current_frame)

        self.current_frame += 1

        return packet

    # ======================================================
    # Read Any Frame
    # ======================================================

    def get_frame(self, index):

        if index < 0 or index >= self.total_frames:
            raise IndexError("Frame index out of range.")

        image = self.dataset.get_cam0(index)

        oxts = self.dataset.oxts[index]

        packet = oxts.packet

        return {

            # --------------------------------------------
            # Metadata
            # --------------------------------------------

            "frame_id": index,

            "timestamp": self.dataset.timestamps[index],

            "image": image,

            # --------------------------------------------
            # GPS
            # --------------------------------------------

            "latitude": packet.lat,

            "longitude": packet.lon,

            "altitude": packet.alt,

            # --------------------------------------------
            # Vehicle Orientation
            # --------------------------------------------

            "roll": packet.roll,

            "pitch": packet.pitch,

            "yaw": packet.yaw,

            # --------------------------------------------
            # Velocity (m/s)
            # --------------------------------------------

            "vf": packet.vf,

            "vl": packet.vl,

            "vu": packet.vu,

            # --------------------------------------------
            # Acceleration (m/s²)
            # --------------------------------------------

            "ax": packet.ax,

            "ay": packet.ay,

            "az": packet.az,

            # --------------------------------------------
            # Angular Velocity (rad/s)
            # --------------------------------------------

            "wx": packet.wx,

            "wy": packet.wy,

            "wz": packet.wz,

            # --------------------------------------------
            # GPS Quality
            # --------------------------------------------

            "numsats": packet.numsats,

            "pos_accuracy": packet.pos_accuracy,

            "vel_accuracy": packet.vel_accuracy,

            "navstat": packet.navstat,

            "posmode": packet.posmode,

            "velmode": packet.velmode,

            "orimode": packet.orimode,

            # --------------------------------------------
            # Raw OXTS
            # --------------------------------------------

            "oxts": oxts,

            "gps": packet,

            "imu": packet
        }

    # ======================================================
    # Jump To Frame
    # ======================================================

    def seek(self, frame_index):

        if frame_index < 0:
            frame_index = 0

        if frame_index >= self.total_frames:
            frame_index = self.total_frames - 1

        self.current_frame = frame_index

    # ======================================================
    # Replay Progress
    # ======================================================

    def progress(self):

        if self.total_frames == 0:
            return 0.0

        return self.current_frame / self.total_frames

    # ======================================================
    # Replay Information
    # ======================================================

    def info(self):

        return {

            "drive": self.drive_name,

            "date": self.date,

            "frames": self.total_frames,

            "current_frame": self.current_frame,

            "progress": round(
                self.progress() * 100,
                2
            )
        }