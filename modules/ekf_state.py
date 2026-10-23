"""
=========================================================
Extended Kalman Filter State
Paper 2

State Vector

X =
[x,
 y,
 heading,
 velocity]

=========================================================
"""

import numpy as np


class EKFState:

    def __init__(self):

        # State vector
        self.x = np.zeros((4, 1), dtype=float)

        # Covariance matrix
        self.P = np.eye(4)

        # Process noise
        self.Q = np.diag([0.05, 0.05, 0.02, 0.10])

    # -------------------------------------------------

    def initialize(self, packet):

        self.x = np.array([
            [packet["longitude"]],
            [packet["latitude"]],
            [packet["yaw"]],
            [packet["vf"]]
        ], dtype=float)

        self.P = np.eye(4)

    # -------------------------------------------------

    def get_state(self):

        return self.x.copy()

    # -------------------------------------------------

    def get_covariance(self):

        return self.P.copy()

    # -------------------------------------------------

    def set_state(self, x):

        self.x = x.copy()

    # -------------------------------------------------

    def set_covariance(self, P):

        self.P = P.copy()