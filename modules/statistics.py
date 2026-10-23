"""
=========================================================
Dataset Statistics Generator
Paper 1

Computes normalization statistics from a KITTI drive or
KITTI Odometry sequence, used to calibrate sensor scoring.
=========================================================
"""

import json
from pathlib import Path

import numpy as np

from modules.replay import KITTIReplay
from modules.odometry_replay import KITTIOdometryReplay
from modules.sensor_metrics import SensorMetrics


class DatasetStatistics:

    def __init__(self, drive=None, *, replay=None):
        """
        Parameters
        ----------
        drive : str, optional
            KITTI Raw drive name (e.g. '2011_09_26_drive_0022_sync').
            Ignored when ``replay`` is supplied.
        replay : KITTIReplay | KITTIOdometryReplay, optional
            Pre-constructed replay object. Takes priority over ``drive``.
        """

        if replay is not None:
            self.replay = replay
        elif drive is not None:
            self.replay = KITTIReplay(drive)
        else:
            raise ValueError("Provide either drive= or replay=.")

        self.metrics = SensorMetrics()

    # -----------------------------------------------------

    def stats_range(self, values):

        values = np.asarray(values)

        return {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "p5": float(np.percentile(values, 5)),
            "p95": float(np.percentile(values, 95))
        }

    # -----------------------------------------------------

    def compute(self):

        self.replay.load()

        blur = []
        brightness = []
        contrast = []
        edge = []

        posacc = []
        velacc = []
        sats = []

        acc = []
        gyro = []
        jerk = []

        while self.replay.has_next():

            packet = self.replay.next_frame()

            m = self.metrics.compute(packet)

            # Camera
            blur.append(m["camera"]["blur"])
            brightness.append(m["camera"]["brightness"])
            contrast.append(m["camera"]["contrast"])
            edge.append(m["camera"]["edge_density"])

            # GPS
            posacc.append(m["gps"]["position_accuracy"])
            velacc.append(m["gps"]["velocity_accuracy"])
            sats.append(m["gps"]["satellites"])

            # IMU
            acc.append(m["imu"]["acceleration"])
            gyro.append(m["imu"]["gyroscope"])
            jerk.append(m["imu"]["jerk"])

        blur_stats = self.stats_range(blur)
        contrast_stats = self.stats_range(contrast)
        edge_stats = self.stats_range(edge)

        posacc_stats = self.stats_range(posacc)
        velacc_stats = self.stats_range(velacc)
        sat_stats = self.stats_range(sats)

        acc_stats = self.stats_range(acc)
        gyro_stats = self.stats_range(gyro)
        jerk_stats = self.stats_range(jerk)

        stats = {

            "blur_min": blur_stats["min"],
            "blur_max": blur_stats["max"],
            "blur_p5": blur_stats["p5"],
            "blur_p95": blur_stats["p95"],

            "contrast_min": contrast_stats["min"],
            "contrast_max": contrast_stats["max"],
            "contrast_p5": contrast_stats["p5"],
            "contrast_p95": contrast_stats["p95"],

            "edge_min": edge_stats["min"],
            "edge_max": edge_stats["max"],
            "edge_p5": edge_stats["p5"],
            "edge_p95": edge_stats["p95"],

            "brightness_mean": float(np.mean(brightness)),
            "brightness_dev_max": float(
                np.max(np.abs(np.array(brightness) - np.mean(brightness)))
            ),

            "posacc_p5": posacc_stats["p5"],
            "posacc_p95": posacc_stats["p95"],

            "velacc_p5": velacc_stats["p5"],
            "velacc_p95": velacc_stats["p95"],

            "sat_p5": sat_stats["p5"],
            "sat_p95": sat_stats["p95"],

            "acc_p5": acc_stats["p5"],
            "acc_p95": acc_stats["p95"],

            "gyro_p5": gyro_stats["p5"],
            "gyro_p95": gyro_stats["p95"],

            "jerk_p5": jerk_stats["p5"],
            "jerk_p95": jerk_stats["p95"],

            # Keep original values for reference
            "sat_min": sat_stats["min"],
            "sat_max": sat_stats["max"]
        }

        return stats

    # -----------------------------------------------------

    def save(self, output_file):

        stats = self.compute()

        output = Path(output_file)

        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, "w") as f:

            json.dump(stats, f, indent=4)

        print()

        print("Statistics saved to:")

        print(output)

        return stats
