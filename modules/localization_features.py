"""
=========================================================
Localization Feature Extraction
Paper 2

Builds the feature vector required by the
Localization Reliability Random Forest model.
=========================================================
"""

import numpy as np
import pandas as pd


class LocalizationFeatures:

    def __init__(self):
        pass

    # -----------------------------------------------------

    def build(self, fusion_result):

        packet = fusion_result["packet"]

        fusion = fusion_result["fusion"]

        quality = fusion_result["quality"]

        weights = fusion_result["weights"]

        innovation = fusion_result["innovation"]

        P = fusion_result["covariance"]["P"]

        imu = packet["imu"]

        features = {

            "velocity": float(fusion["velocity"]),

            "yaw": float(fusion["yaw"]),

            "acceleration": float(imu.ax),

            "yaw_rate": float(imu.wz),

            "camera_weight": float(weights["camera"]),

            "gps_weight": float(weights["gps"]),

            "imu_weight": float(weights["imu"]),

            "innovation_norm": float(np.linalg.norm(innovation)),

            "covariance_trace": float(np.trace(P))

        }

        return features

    # -----------------------------------------------------

    def to_dataframe(self, features):

        return pd.DataFrame([{

            "Velocity": features["velocity"],

            "Heading": features["yaw"],

            "Acceleration": features["acceleration"],

            "Yaw Rate": features["yaw_rate"],

            "Camera Weight": features["camera_weight"],

            "GPS Weight": features["gps_weight"],

            "IMU Weight": features["imu_weight"],

            "Innovation Norm": features["innovation_norm"],

            "Covariance Trace": features["covariance_trace"]

        }])

    # -----------------------------------------------------

    def to_numpy(self, features):

        return self.to_dataframe(features).to_numpy(dtype=np.float32)