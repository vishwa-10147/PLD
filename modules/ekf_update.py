"""
=========================================================
Adaptive Extended Kalman Filter Update
Paper 2

Performs adaptive measurement update using
confidence-driven covariance matrices.
=========================================================
"""

import numpy as np


class EKFUpdate:

    def __init__(self):
        pass

    # -------------------------------------------------

    def measurement_vector(self, packet):

        return np.array([
            [packet["longitude"]],
            [packet["latitude"]],
            [packet["yaw"]],
            [packet["vf"]]
        ], dtype=float)

    # -------------------------------------------------

    def measurement_matrix(self):

        # Identity because we directly observe all states
        return np.eye(4)

    # -------------------------------------------------

    def build_measurement_covariance(self, adaptive_covariance):

        """
        Expand the 3x3 adaptive covariance matrix
        (Camera, GPS, IMU)

        into the 4x4 EKF measurement covariance.

        State:
        longitude
        latitude
        yaw
        velocity
        """

        R3 = adaptive_covariance

        gps_noise = R3[1, 1]
        imu_noise = R3[2, 2]

        R = np.diag([

            gps_noise,
            gps_noise,
            imu_noise,
            imu_noise

        ])

        return R

    # -------------------------------------------------

    def update(self, state, packet, adaptive_covariance):

        x = state.get_state()
        P = state.get_covariance()

        z = self.measurement_vector(packet)

        H = self.measurement_matrix()

        R = self.build_measurement_covariance(adaptive_covariance)

        # Innovation
        y = z - H @ x

        # Innovation covariance
        S = H @ P @ H.T + R

        # Kalman Gain
        K = P @ H.T @ np.linalg.inv(S)

        # Updated state
        x_new = x + K @ y

        # Updated covariance
        I = np.eye(4)

        P_new = (I - K @ H) @ P

        state.set_state(x_new)
        state.set_covariance(P_new)

        return {

            "state": x_new,

            "covariance": P_new,

            "kalman_gain": K,

            "innovation": y,

            "measurement_covariance": R

        }