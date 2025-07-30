from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import xgboost as xgb
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder


class XGBoostPlanner:
    """
    A planner using a collection of XGBoost models to predict next states.
    It uses scikit-learn's MultiOutputClassifier for convenience.
    """

    def __init__(self, encoding_type="bin", num_blocks: Optional[int] = None, seed=42):
        self.encoding_type = encoding_type
        self.num_blocks = num_blocks
        self.seed = seed
        self.model = None
        self.label_encoders = None  # For SAS+ encoding

        if self.encoding_type == "bin":
            # Multi-label classification problem
            xgb_estimator = xgb.XGBClassifier(  # type: ignore
                objective="binary:logistic", eval_metric="logloss", random_state=self.seed
            )
            self.model = MultiOutputClassifier(estimator=xgb_estimator, n_jobs=-1)
        elif self.encoding_type == "sas":
            if num_blocks is None:
                raise ValueError("num_blocks must be provided for SAS+ encoding.")
            # Multi-output classification problem
            xgb_estimator = xgb.XGBClassifier(  # type: ignore
                objective="multi:softprob", eval_metric="mlogloss", random_state=self.seed
            )
            self.model = MultiOutputClassifier(estimator=xgb_estimator, n_jobs=-1)
            self.label_encoders = [LabelEncoder() for _ in range(num_blocks)]
        else:
            raise ValueError(f"Unsupported encoding type: {self.encoding_type}")

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Trains the multi-output XGBoost model.
        """
        print(f"Training XGBoost model on data with shape X: {X_train.shape}, y: {y_train.shape}")

        if self.model is None:
            raise RuntimeError("Model must be initialized before training.")

        if self.encoding_type == "sas":
            y_transformed = np.zeros_like(y_train)

            if self.num_blocks is None:
                raise ValueError("num_blocks must be set for SAS+ encoding.")
            if self.label_encoders is None:
                raise ValueError("label_encoders must be initialized for SAS+ encoding.")

            for i in range(self.num_blocks):
                y_transformed[:, i] = self.label_encoders[i].fit_transform(y_train[:, i])
            self.model.fit(X_train, y_transformed)

        else:  # binary
            self.model.fit(X_train, y_train)

        print("XGBoost model training complete.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the next state(s).
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained or loaded.")

        y_pred = self.model.predict(X)

        if self.encoding_type == "sas":
            y_pred_original = np.zeros_like(y_pred)

            if self.num_blocks is None:
                raise ValueError("num_blocks must be set for SAS+ encoding.")
            if self.label_encoders is None:
                raise ValueError("label_encoders must be initialized for SAS+ encoding.")

            for i in range(self.num_blocks):
                y_pred_original[:, i] = self.label_encoders[i].inverse_transform(y_pred[:, i])  # type: ignore

            return y_pred_original.astype(int)

        return y_pred.astype(int)  # type: ignore

    def save(self, path: Path):
        """Saves the trained model and label encoders to a file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        save_data = {"model": self.model, "label_encoders": self.label_encoders}
        joblib.dump(save_data, path)
        print(f"XGBoost model saved to {path}")

    @classmethod
    def load(cls, path: Path, encoding_type: str, num_blocks: int, seed: int) -> "XGBoostPlanner":
        """Loads a model from a file."""
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found at {path}")

        instance = cls(encoding_type=encoding_type, num_blocks=num_blocks, seed=seed)

        load_data = joblib.load(path)
        instance.model = load_data["model"]
        instance.label_encoders = load_data["label_encoders"]

        print(f"XGBoost model loaded from {path}")
        return instance
