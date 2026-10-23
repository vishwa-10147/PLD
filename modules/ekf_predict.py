"""
=========================================================
Extended Kalman Filter Prediction
Paper 2
=========================================================
"""

import numpy as np


class EKFPredict:

    def __init__(self, dt=0.1):
        self.dt = dt

    # -------------------------------------------------

    def state_transition(self, state):

        x = state.get_state()

        lon = x[0, 0]
        lat = x[1, 0]
        yaw = x[2, 0]
        velocity = x[3, 0]

        dt = self.dt

        # Constant Velocity Motion Model
        predicted = np.array([
            [lon + velocity * np.cos(yaw) * dt],
            [lat + velocity * np.sin(yaw) * dt],
            [yaw],
            [velocity]
        ])

        return predicted

    # -------------------------------------------------

    def transition_matrix(self, state):

        x = state.get_state()

        yaw = x[2, 0]
        velocity = x[3, 0]

        dt = self.dt

        F = np.array([

            [1, 0, -velocity * np.sin(yaw) * dt,
                 np.cos(yaw) * dt],

            [0, 1, velocity * np.cos(yaw) * dt,
                 np.sin(yaw) * dt],

            [0, 0, 1, 0],

            [0, 0, 0, 1]

        ])

        return F

    # -------------------------------------------------

    def predict(self, state):

        x = state.get_state()
        P = state.get_covariance()

        F = self.transition_matrix(state)

        x_pred = self.state_transition(state)

        P_pred = F @ P @ F.T + state.Q

        state.set_state(x_pred)
        state.set_covariance(P_pred)

        return {

            "state": x_pred,

            "covariance": P_pred

        }