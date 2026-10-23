"""
=========================================================
Sensor Metric Extraction Module
Paper 1
Confidence-Driven Predictive Localization Dashboard
=========================================================

Extracts raw sensor quality metrics from

1. Camera
2. GPS
3. IMU

These metrics are later converted into
Sensor Confidence Index (SCI).
"""

import cv2
import numpy as np


# ==========================================================
# CAMERA
# ==========================================================

class CameraMetrics:

    def compute(self, image):

        # Convert PIL image -> OpenCV

        img = np.array(image)

        img = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2BGR
        )

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        # -----------------------------------------
        # Blur
        # -----------------------------------------

        blur = cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()

        # -----------------------------------------
        # Brightness
        # -----------------------------------------

        brightness = gray.mean()

        # -----------------------------------------
        # Contrast
        # -----------------------------------------

        contrast = gray.std()

        # -----------------------------------------
        # Edge Density
        # -----------------------------------------

        edges = cv2.Canny(
            gray,
            100,
            200
        )

        edge_density = (
            np.count_nonzero(edges)
            / edges.size
        )

        return {

            "blur": float(blur),

            "brightness": float(brightness),

            "contrast": float(contrast),

            "edge_density": float(edge_density)

        }


# ==========================================================
# GPS
# ==========================================================

class GPSMetrics:

    def compute(self, packet):

        return {

            "position_accuracy":

                float(packet["pos_accuracy"]),

            "velocity_accuracy":

                float(packet["vel_accuracy"]),

            "satellites":

                int(packet["numsats"]),

            "navstat":

                int(packet["navstat"]),

            "posmode":

                int(packet["posmode"]),

            "velmode":

                int(packet["velmode"])

        }


# ==========================================================
# IMU
# ==========================================================

class IMUMetrics:

    def __init__(self):

        self.previous_acc = None

    def compute(self, packet):

        acc = np.array([

            packet["ax"],

            packet["ay"],

            packet["az"]

        ])

        gyro = np.array([

            packet["wx"],

            packet["wy"],

            packet["wz"]

        ])

        acc_mag = np.linalg.norm(acc)

        gyro_mag = np.linalg.norm(gyro)

        if self.previous_acc is None:

            jerk = 0.0

        else:

            jerk = np.linalg.norm(

                acc - self.previous_acc

            )

        self.previous_acc = acc.copy()

        return {

            "acceleration":

                float(acc_mag),

            "gyroscope":

                float(gyro_mag),

            "jerk":

                float(jerk)

        }


# ==========================================================
# ALL SENSOR METRICS
# ==========================================================

class SensorMetrics:

    def __init__(self):

        self.camera = CameraMetrics()

        self.gps = GPSMetrics()

        self.imu = IMUMetrics()

    def compute(self, replay_packet):

        camera_metrics = self.camera.compute(

            replay_packet["image"]

        )

        gps_metrics = self.gps.compute(

            replay_packet

        )

        imu_metrics = self.imu.compute(

            replay_packet

        )

        return {

            "camera": camera_metrics,

            "gps": gps_metrics,

            "imu": imu_metrics

        }