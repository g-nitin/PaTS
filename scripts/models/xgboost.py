from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import xgboost as xgb
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder

from ..pats_dataset import PaTSDataset


def prepare_data_for_xgboost(dataset: PaTSDataset, context_window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flattens the sequential dataset into a tabular format (X, y) for XGBoost, including a context window of past states.
    X = (S_{t-k+1}, ..., S_t, S_G)
    y = S_{t+1}
    """
    X_list, y_list = [], []
    for i in range(len(dataset)):
        item = dataset[i]
        trajectory = item["expert_trajectory"]  # (L, F)
        goal_state = item["goal_state"]  # (F,)
        initial_state = item["initial_state"]  # (F,)

        # Create (S_{t-k+1}, ..., S_t, S_G) -> S_{t+1} pairs
        for t in range(len(trajectory) - 1):  # Iterate from S_0 to S_{L-2} to predict S_1 to S_{L-1}
            # current_state = trajectory[t]
            next_state = trajectory[t + 1]

            # Build the context window for S_t
            context_states = []
            for j in range(context_window_size):
                idx_in_traj = t - (context_window_size - 1 - j)
                if idx_in_traj >= 0:
                    context_states.append(trajectory[idx_in_traj])
                else:
                    # Pad with initial_state if not enough history
                    context_states.append(initial_state)

            # Concatenate context states and goal state to form the feature vector X
            X_sample = np.concatenate(context_states + [goal_state])
            X_list.append(X_sample)
            y_list.append(next_state)

    return np.array(X_list), np.array(y_list)


class XGBoostPlanner:
    """
    A planner using a collection of XGBoost models to predict next states.
    It uses scikit-learn's MultiOutputClassifier for convenience.
    """

    def __init__(self, encoding_type="bin", num_blocks: Optional[int] = None, seed=42, context_window_size: int = 1):
        self.encoding_type = encoding_type
        self.num_blocks = num_blocks
        self.seed = seed
        self.model = None
        self.label_encoders = None  # For SAS+ encoding

        if self.encoding_type == "bin":
            self.context_window_size = context_window_size
            # Multi-label classification problem
            xgb_estimator = xgb.XGBClassifier(  # type: ignore
                objective="binary:logistic", eval_metric="logloss", random_state=self.seed
            )
            self.model = MultiOutputClassifier(estimator=xgb_estimator, n_jobs=-1)
        elif self.encoding_type == "sas":
            if num_blocks is None:
                raise ValueError("num_blocks must be provided for SAS+ encoding.")
            self.context_window_size = context_window_size
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
        save_data = {
            "model": self.model,
            "label_encoders": self.label_encoders,
            "encoding_type": self.encoding_type,  # Save these for robust loading
            "num_blocks": self.num_blocks,  # Save these for robust loading
            "context_window_size": self.context_window_size,  # Save new attribute
        }
        joblib.dump(save_data, path)
        print(f"XGBoost model saved to {path}")

    @classmethod
    def load(cls, path: Path, encoding_type: str, num_blocks: int, seed: int):
        """Loads a model from a file."""
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found at {path}")

        load_data = joblib.load(path)

        # Retrieve parameters from saved data for initialization
        # This makes loading more robust, as the init parameters are derived from the saved model
        loaded_encoding_type = load_data.get("encoding_type", encoding_type)  # Fallback to passed arg for old models
        loaded_num_blocks = load_data.get("num_blocks", num_blocks)  # Fallback to passed arg for old models
        loaded_context_window_size = load_data.get("context_window_size", 1)  # Fallback to default for old models

        instance = cls(
            encoding_type=loaded_encoding_type,
            num_blocks=loaded_num_blocks,
            seed=seed,  # Seed is not saved, always passed
            context_window_size=loaded_context_window_size,
        )

        instance.model = load_data["model"]
        instance.label_encoders = load_data["label_encoders"]

        print(f"XGBoost model loaded from {path}")
        return instance
