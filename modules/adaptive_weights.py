"""
===============================================
Adaptive Sensor Weight Estimator
Paper 2

Converts Paper 1 confidence scores into
adaptive fusion weights.
===============================================
"""


class AdaptiveWeights:

    def __init__(self):

        pass

    def compute(self, quality):

        camera = max(quality["camera_score"], 1e-6)
        gps = max(quality["gps_score"], 1e-6)
        imu = max(quality["imu_score"], 1e-6)

        total = camera + gps + imu

        weights = {

            "camera": camera / total,

            "gps": gps / total,

            "imu": imu / total

        }

        return weights