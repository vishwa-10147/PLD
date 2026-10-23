"""
=========================================================
Localization Reliability Prediction
Paper 2

Loads the trained Random Forest model and predicts
vehicle localization reliability.

Author : Vishwa
Paper  : Confidence-Driven Adaptive Multi-Sensor Fusion
=========================================================
"""

from pathlib import Path

import joblib
import numpy as np


class LocalizationModel:

    def __init__(self,
                 model_path="models/localization_reliability_rf.joblib"):

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Localization model not found:\n{self.model_path.resolve()}"
            )

        self.model = joblib.load(self.model_path)

        print(f"✓ Localization Model Loaded")
        print(f"  Model : {self.model_path.name}")

    # --------------------------------------------------

    def predict(self, feature_vector):

        """
        Predict localization reliability.

        Parameters
        ----------
        feature_vector :
            Pandas DataFrame or NumPy array

        Returns
        -------
        dict
        """

        prediction = self.model.predict(feature_vector)

        score = float(prediction[0])

        # Clamp between 0 and 1
        score = np.clip(score, 0.0, 1.0)

        percentage = score * 100

        # -------------------------------
        # Reliability Category
        # -------------------------------

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

            "score": round(score, 4),

            "percentage": round(percentage, 2),

            "status": status,

            "color": color

        }

    # --------------------------------------------------

    def __call__(self, feature_vector):
        """
        Allows:

        prediction = model(features)
        """

        return self.predict(feature_vector)