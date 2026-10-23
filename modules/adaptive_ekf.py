"""
=================================================
Adaptive Extended Kalman Filter
Paper 2

Uses adaptive sensor weights to build
dynamic measurement covariance matrices.
=================================================
"""

import numpy as np


class AdaptiveEKF:

    def __init__(self):

        # Base measurement variances
        self.base_camera = 1.0
        self.base_gps = 1.0
        self.base_imu = 1.0

    # -------------------------------------------------

    def measurement_covariance(self, weights):

        camera_noise = self.base_camera / max(weights["camera"], 1e-6)

        gps_noise = self.base_gps / max(weights["gps"], 1e-6)

        imu_noise = self.base_imu / max(weights["imu"], 1e-6)

        R = np.diag([

            camera_noise,

            gps_noise,

            imu_noise

        ])

        return R

    # -------------------------------------------------

    def process_covariance(self):

        Q = np.diag([

            0.05,

            0.05,

            0.02

        ])

        return Q

    # -------------------------------------------------

    def compute(self, weights):

        return {

            "R": self.measurement_covariance(weights),

            "Q": self.process_covariance()

        }