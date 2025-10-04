import warnings
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor

from ..pats_dataset import PaTSDataset


def prepare_data_for_xgboost(
    dataset: PaTSDataset, context_window_size: int, max_plan_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flattens the dataset into a tabular format (X, y) for direct multi-step forecasting.
    X = flatten(S_{t-k+1}, ..., S_t, S_G)
    y = flatten(S_{t+1}, S_{t+2}, ..., S_Goal, <pad>, ..., <pad>)

    The recommended setting for `max_plan_length` should be generous enough for the domain (e.g., 60 for 4 blocks).
    """
    X_list, y_list = [], []
    state_dim = dataset.state_dim

    # Using a padding value that is unlikely to be a valid state feature, e.g., -99
    # For SAS+, valid values are -1 to N. For binary, 0 or 1. -99 works for both.
    padding_value = -99

    for i in range(len(dataset)):
        item = dataset[i]
        trajectory = item["expert_trajectory"]  # (L, F)
        goal_state = item["goal_state"]  # (F,)
        initial_state = item["initial_state"]  # (F,)

        # We only need one training instance per plan: (S_0, S_G) -> (S_1, ..., S_Goal)
        # Previous works (1) creates a sample from every possible window, but for planning, starting from S_0 is the most critical task.
        # (1) Elsayed, Shereen, et al. "Do we really need deep learning models for time series forecasting?." arXiv preprint arXiv:2101.02118 (2021).

        # Build the context window for the initial state S_0
        context_states = [initial_state] * context_window_size

        # Create the input vector X
        X_sample = np.concatenate(context_states + [goal_state])
        X_list.append(X_sample)

        # Create the output vector Y
        remaining_plan = trajectory[1:]  # S_1, S_2, ... S_Goal

        # Pad the remaining plan to max_plan_length
        num_states_to_pad = max_plan_length - len(remaining_plan)

        if num_states_to_pad < 0:
            # If the expert plan is longer than our max, truncate it.
            remaining_plan = remaining_plan[:max_plan_length]
            padded_plan = remaining_plan
            # Warn the user
            warnings.warn(
                f"Expert plan length {len(trajectory) - 1} exceeds max_plan_length {max_plan_length}. Truncating the plan."
            )
        else:
            padding = np.full((num_states_to_pad, state_dim), padding_value, dtype=remaining_plan.dtype)
            padded_plan = np.vstack([remaining_plan, padding]) if len(remaining_plan) > 0 else padding

        y_list.append(padded_plan.flatten())

    return np.array(X_list), np.array(y_list)


class XGBoostPlanner:
    """
    A planner using a collection of XGBoost models for direct multi-step REGRESSION.
    """

    def __init__(
        self,
        encoding_type="bin",
        num_blocks: Optional[int] = None,
        seed=42,
        context_window_size: int = 1,
        max_plan_length: Optional[int] = None,
    ):
        self.encoding_type = encoding_type
        self.num_blocks = num_blocks
        self.seed = seed
        self.context_window_size = context_window_size
        self.max_plan_length = max_plan_length

        # Use XGBRegressor for both binary (0/1) and SAS+ (integer) targets.
        xgb_estimator = xgb.XGBRegressor(objective="reg:squarederror", eval_metric="rmse", random_state=self.seed)
        self.model = MultiOutputRegressor(estimator=xgb_estimator, n_jobs=-1)

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Trains the multi-output XGBoost regression model.
        """
        print(f"Training XGBoost REGRESSION model on data with shape X: {X_train.shape}, y: {y_train.shape}")
        if self.model is None:
            raise RuntimeError("Model must be initialized before training.")

        # No special label handling is needed for regression.
        self.model.fit(X_train, y_train)

        print("XGBoost model training complete.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the next state(s) and rounds the result to the nearest integer.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained or loaded.")

        # Predict will return floating point values
        y_pred_float = self.model.predict(X)

        # Round to the nearest integer and cast to int type
        y_pred_rounded = np.round(y_pred_float).astype(int)  # type: ignore

        return y_pred_rounded

    def save(self, path: Path):
        """Saves the trained model to a file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        save_data = {
            "model": self.model,
            "encoding_type": self.encoding_type,
            "num_blocks": self.num_blocks,
            "context_window_size": self.context_window_size,
            "max_plan_length": self.max_plan_length,
        }
        joblib.dump(save_data, path)
        print(f"XGBoost model saved to {path}")

    @classmethod
    def load(cls, path: Path, encoding_type: str, num_blocks: int, seed: int):
        """Loads a model from a file."""
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found at {path}")
        load_data = joblib.load(path)

        loaded_encoding_type = load_data.get("encoding_type", encoding_type)
        loaded_num_blocks = load_data.get("num_blocks", num_blocks)
        loaded_context_window_size = load_data.get("context_window_size", 1)
        loaded_max_plan_length = load_data.get("max_plan_length")

        instance = cls(
            encoding_type=loaded_encoding_type,
            num_blocks=loaded_num_blocks,
            seed=seed,
            context_window_size=loaded_context_window_size,
            max_plan_length=loaded_max_plan_length,
        )
        instance.model = load_data["model"]

        print(f"XGBoost model loaded from {path}")
        return instance
