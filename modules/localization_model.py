"""
=========================================================
Localization Reliability Prediction
Paper 2

Loads trained Random Forest model and predicts
vehicle localization reliability.
=========================================================
"""

from pathlib import Path

import joblib
import numpy as np


class LocalizationModel:

    def __init__(
        self,
        model_path="models/localization_reliability_rf.joblib"
    ):

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Localization model not found:\n{self.model_path.resolve()}"
            )

        self.model = joblib.load(self.model_path)

        print("\n✓ Localization Model Loaded")
        print(f"Model : {self.model_path.name}")

    # -----------------------------------------------------

    def predict(self, feature_vector):

        prediction = self.model.predict(feature_vector)

        score = float(prediction[0])

        score = max(0.0, min(1.0, score))

        percentage = score * 100.0

        if score >= 0.85:
            status = "HIGH"
            color = "green"

        elif score >= 0.60:
            status = "MEDIUM"
            color = "orange"

        else:
            status = "LOW"
            color = "red"

        return {

            "score": float(round(score, 4)),

            "percentage": float(round(percentage, 2)),

            "status": status,

            "color": color

        }

    # -----------------------------------------------------

    def __call__(self, feature_vector):

        return self.predict(feature_vector)