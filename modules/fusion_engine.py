"""
=========================================================
Confidence-Driven Adaptive Sensor Fusion Engine
Paper 2

Complete Fusion Pipeline

KITTI Replay
        │
        ▼
Sensor Metrics
        │
        ▼
Sensor Confidence Index (SCI)
        │
        ▼
Adaptive Sensor Weights
        │
        ▼
Adaptive Covariance
        │
        ▼
Adaptive EKF
        │
        ▼
Localization Feature Extraction
        │
        ▼
Random Forest Localization Reliability
=========================================================
"""

from modules.sensor_metrics import SensorMetrics
from modules.sensor_quality import SensorConfidence

from modules.adaptive_weights import AdaptiveWeights
from modules.adaptive_ekf import AdaptiveEKF

from modules.ekf_state import EKFState
from modules.ekf_predict import EKFPredict
from modules.ekf_update import EKFUpdate

from modules.localization_features import LocalizationFeatures
from modules.localization_model import LocalizationModel


class FusionEngine:

    def __init__(self, stats_file=None):

        # -------------------------------------------------
        # Paper 1
        # -------------------------------------------------

        self.metrics = SensorMetrics()
        self.quality = SensorConfidence(stats_file=stats_file)

        # -------------------------------------------------
        # Paper 2
        # -------------------------------------------------

        self.weights = AdaptiveWeights()
        self.adaptive = AdaptiveEKF()

        self.state = EKFState()
        self.predictor = EKFPredict(dt=0.1)
        self.updater = EKFUpdate()

        # Localization Reliability

        self.feature_builder = LocalizationFeatures()
        self.localization_model = LocalizationModel()

        self.initialized = False

    # ---------------------------------------------------------

    def process(self, packet):

        # ------------------------------------------
        # Initialize EKF
        # ------------------------------------------

        if not self.initialized:

            self.state.initialize(packet)
            self.initialized = True

        # ------------------------------------------
        # Paper 1
        # Sensor Metrics
        # ------------------------------------------

        metrics = self.metrics.compute(packet)

        quality = self.quality.compute(metrics)

        # ------------------------------------------
        # Adaptive Sensor Weights
        # ------------------------------------------

        weights = self.weights.compute(quality)

        # ------------------------------------------
        # Adaptive Covariance
        # ------------------------------------------

        covariance = self.adaptive.compute(weights)

        # ------------------------------------------
        # EKF Prediction
        # ------------------------------------------

        self.predictor.predict(self.state)

        # ------------------------------------------
        # EKF Update
        # ------------------------------------------

        result = self.updater.update(

            self.state,
            packet,
            covariance["R"]

        )

        covariance["P"] = result["covariance"]

        state = result["state"]

        # ------------------------------------------
        # Final Fused Localization
        # ------------------------------------------

        fused = {

            "longitude": float(state[0, 0]),

            "latitude": float(state[1, 0]),

            "yaw": float(state[2, 0]),

            "velocity": float(state[3, 0])

        }

        # ------------------------------------------
        # Temporary Fusion Result
        # ------------------------------------------

        fusion_result = {

            "packet": packet,

            "metrics": metrics,

            "quality": quality,

            "weights": weights,

            "covariance": covariance,

            "fusion": fused,

            "kalman_gain": result["kalman_gain"],

            "innovation": result["innovation"]

        }

        # ------------------------------------------
        # Localization Features
        # ------------------------------------------

        features = self.feature_builder.build(fusion_result)

        feature_df = self.feature_builder.to_dataframe(features)

        # ------------------------------------------
        # Random Forest Prediction
        # ------------------------------------------

        localization = self.localization_model.predict(feature_df)

        # ------------------------------------------
        # Final Output
        # ------------------------------------------

        return {

            "packet": packet,

            "metrics": metrics,

            "quality": quality,

            "weights": weights,

            "covariance": covariance,

            "fusion": fused,

            "features": features,

            "localization": localization,

            "kalman_gain": result["kalman_gain"],

            "innovation": result["innovation"]

        }