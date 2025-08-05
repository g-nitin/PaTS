from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List

import numpy as np
import torch


# ** Abstract Plannable Model **
class PlannableModel(ABC):
    def __init__(self, model_path: Path, num_blocks: int, device: torch.device):
        self.model_path = model_path
        self.num_blocks = num_blocks
        self.device = device
        self.model: Any = None  # To be initialized by load_model

    @abstractmethod
    def load_model(self):
        """Loads the model from self.model_path."""
        pass

    @abstractmethod
    def predict_sequence(self, initial_state_np: np.ndarray, goal_state_np: np.ndarray, max_length: int) -> List[List[int]]:
        """
        Predicts a sequence of states (plan).
        Inputs are 0/1 numpy arrays. Output is List of 0/1 state lists.
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Returns a descriptive name of the model."""
        pass

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Returns the feature dimension of the states the model expects/produces."""
        pass
