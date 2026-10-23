"""
=========================================================
Sensor Confidence Module
Paper 1

Computes:

1. Camera Quality Score
2. GPS Quality Score
3. IMU Quality Score
4. Sensor Confidence Index (SCI)

Uses statistics generated from statistics.py
=========================================================
"""

import json
from pathlib import Path


class SensorConfidence:

    def __init__(self, stats_file=None):
        """
        Parameters
        ----------
        stats_file : str | Path | None
            Path to the JSON statistics file.
            Defaults to ``outputs/statistics/kitti_stats.json``
            (the original KITTI Raw calibration).
        """

        if stats_file is None:
            stats_file = Path("outputs/statistics/kitti_stats.json")
        else:
            stats_file = Path(stats_file)

        if not stats_file.exists():
            raise FileNotFoundError(
                f"Statistics file not found:\n{stats_file}\n"
                "Run the appropriate statistics generator first."
            )

        with open(stats_file, "r") as f:
            self.stats = json.load(f)

        # ----------------------------
        # Paper 1 Weights
        # ----------------------------

        self.camera_weights = {
            "blur": 0.35,
            "brightness": 0.20,
            "contrast": 0.20,
            "edge_density": 0.25
        }

        self.gps_weights = {
            "position_accuracy": 0.30,
            "velocity_accuracy": 0.25,
            "satellites": 0.20,
            "posmode": 0.15,
            "velmode": 0.10
        }

        self.imu_weights = {
            "acceleration": 0.40,
            "gyroscope": 0.35,
            "jerk": 0.25
        }

    # =====================================================
    # ROBUST NORMALIZATION
    # =====================================================

    @staticmethod
    def normalize_high(value, p5, p95):

        if p95 <= p5:
            return 1.0

        score = (value - p5) / (p95 - p5)

        return max(0.0, min(1.0, score))

    @staticmethod
    def normalize_low(value, p5, p95):

        return 1.0 - SensorConfidence.normalize_high(
            value,
            p5,
            p95
        )

    # =====================================================
    # CAMERA
    # =====================================================

    def camera_score(self, m):

        blur = self.normalize_high(
            m["blur"],
            self.stats["blur_p5"],
            self.stats["blur_p95"]
        )

        contrast = self.normalize_high(
            m["contrast"],
            self.stats["contrast_p5"],
            self.stats["contrast_p95"]
        )

        edge = self.normalize_high(
            m["edge_density"],
            self.stats["edge_p5"],
            self.stats["edge_p95"]
        )

        brightness = 1.0 - abs(
            m["brightness"] -
            self.stats["brightness_mean"]
        ) / self.stats["brightness_dev_max"]

        brightness = max(0.0, min(1.0, brightness))

        score = (

            self.camera_weights["blur"] * blur +

            self.camera_weights["brightness"] * brightness +

            self.camera_weights["contrast"] * contrast +

            self.camera_weights["edge_density"] * edge

        )

        return score

    # =====================================================
    # GPS
    # =====================================================

    def gps_score(self, m):

        # -----------------------------------------
        # Position Accuracy (meters)
        # Lower is better
        # -----------------------------------------

        pa = m["position_accuracy"]

        if pa <= 0.5:
            pos = 1.0
        elif pa <= 1.0:
            pos = 0.9
        elif pa <= 2.0:
            pos = 0.8
        elif pa <= 5.0:
            pos = 0.6
        else:
            pos = 0.3

        # -----------------------------------------
        # Velocity Accuracy (m/s)
        # Lower is better
        # -----------------------------------------

        va = m["velocity_accuracy"]

        if va <= 0.03:
            vel = 1.0
        elif va <= 0.05:
            vel = 0.9
        elif va <= 0.10:
            vel = 0.8
        elif va <= 0.20:
            vel = 0.6
        else:
            vel = 0.3

        # -----------------------------------------
        # Number of Satellites
        # -----------------------------------------

        sats = m["satellites"]

        if sats >= 12:
            sat = 1.0
        elif sats >= 10:
            sat = 0.9
        elif sats >= 8:
            sat = 0.8
        elif sats >= 6:
            sat = 0.6
        else:
            sat = 0.3

        # -----------------------------------------
        # GNSS Modes
        # -----------------------------------------

        posmode = min(m["posmode"] / 5.0, 1.0)
        velmode = min(m["velmode"] / 5.0, 1.0)

        score = (

            self.gps_weights["position_accuracy"] * pos +

            self.gps_weights["velocity_accuracy"] * vel +

            self.gps_weights["satellites"] * sat +

            self.gps_weights["posmode"] * posmode +

            self.gps_weights["velmode"] * velmode

        )

        return score

    # =====================================================
    # IMU
    # =====================================================

    def imu_score(self, m):

        acc = self.normalize_low(
            m["acceleration"],
            self.stats["acc_p5"],
            self.stats["acc_p95"]
        )

        gyro = self.normalize_low(
            m["gyroscope"],
            self.stats["gyro_p5"],
            self.stats["gyro_p95"]
        )

        jerk = self.normalize_low(
            m["jerk"],
            self.stats["jerk_p5"],
            self.stats["jerk_p95"]
        )

        score = (

            self.imu_weights["acceleration"] * acc +

            self.imu_weights["gyroscope"] * gyro +

            self.imu_weights["jerk"] * jerk

        )

        return score

    # =====================================================
    # SCI
    # =====================================================

    def compute(self, metrics):

        camera = self.camera_score(metrics["camera"])

        gps = self.gps_score(metrics["gps"])

        imu = self.imu_score(metrics["imu"])

        sci = (camera + gps + imu) / 3.0

        if sci >= 0.85:
            status = "EXCELLENT"
        elif sci >= 0.70:
            status = "GOOD"
        elif sci >= 0.50:
            status = "FAIR"
        else:
            status = "POOR"

        return {

            "camera_score": round(camera, 3),

            "gps_score": round(gps, 3),

            "imu_score": round(imu, 3),

            "sci": round(sci, 3),

            "status": status
        }
