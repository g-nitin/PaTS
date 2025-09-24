from pathlib import Path
from typing import Any, Dict

import numpy as np
from torch.utils.data import Dataset


class PaTSDataset(Dataset):
    """
    A PyTorch Dataset for loading Planning as Time-Series (PaTS) data.
    It loads pre-encoded binary trajectories and goal states from .npy files.
    """

    def __init__(
        self, raw_data_dir: str | Path, processed_data_dir: str | Path, split_file_name: str, encoding_type: str = "bin"
    ):
        """
        Initializes the PaTSDataset.

        :param raw_data_dir: The root directory for raw problem data for a specific N (e.g., 'data/raw_problems/blocksworld/N4/'). This is where split files (train_files.txt) and encoding_info.json are located.
        :param processed_data_dir: The root directory for processed, encoded trajectories for a specific N and encoding (e.g., 'data/processed_trajectories/blocksworld/N4/bin/').
        :param split_file_name: The name of the file containing problem basenames for this split (e.g., 'train_files.txt').
        :param encoding_type: The encoding of the data to load ('bin' or 'sas').
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.split_file_path = self.raw_data_dir / "splits" / split_file_name  # Split files are now in a 'splits' subdir
        self.encoding_type = encoding_type

        self.basenames = self._load_basenames()
        if not self.basenames:
            raise ValueError(
                f"No basenames loaded from {self.split_file_path}. "
                "Ensure the file exists, is not empty, and is in the correct directory."
            )

        self.state_dim = self._infer_state_dim()
        if self.state_dim is None or self.state_dim <= 0:
            raise ValueError(f"Inferred state_dim is {self.state_dim}, which is invalid.")

    def _load_basenames(self) -> list[str]:
        if not self.split_file_path.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_file_path}")
        with open(self.split_file_path, "r") as f:
            basenames = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        return basenames

    def _infer_state_dim(self) -> int:
        # Try to load the first trajectory to get state_dim
        # This assumes all trajectories in the dataset share the same state dimension.
        if not self.basenames:
            # This should have been caught by the check in __init__ after _load_basenames
            raise RuntimeError("Cannot infer state_dim: no basenames available.")

        first_basename = self.basenames[0]
        traj_path = self.processed_data_dir / f"{first_basename}.traj.{self.encoding_type}.npy"
        if not traj_path.exists():
            raise FileNotFoundError(
                f"Trajectory file for state_dim inference not found: {traj_path}. Checked for basename: {first_basename}"
            )

        try:
            trajectory_np = np.load(traj_path)
            if trajectory_np.ndim == 2 and trajectory_np.shape[0] > 0:
                return trajectory_np.shape[1]
            else:
                raise ValueError(
                    f"Trajectory file {traj_path} is empty or malformed for state_dim inference. "
                    f"Shape: {trajectory_np.shape}"
                )
        except Exception as e:
            raise IOError(f"Error loading trajectory {traj_path} for state_dim inference: {e}") from e

    def __len__(self) -> int:
        return len(self.basenames)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        basename = self.basenames[idx]

        traj_path = self.processed_data_dir / f"{basename}.traj.{self.encoding_type}.npy"
        goal_path = self.processed_data_dir / f"{basename}.goal.{self.encoding_type}.npy"

        if not traj_path.exists():
            raise FileNotFoundError(f"Trajectory file not found for basename {basename}: {traj_path}")
        if not goal_path.exists():
            raise FileNotFoundError(f"Goal file not found for basename {basename}: {goal_path}")

        try:
            # Load trajectory and goal based on encoding type
            if self.encoding_type == "sas":
                # For SAS+, we need integer types for embedding layers and loss functions
                expert_trajectory_np = np.load(traj_path).astype(np.int64)
                goal_state_np = np.load(goal_path).astype(np.int64)
            else:  # binary
                expert_trajectory_np = np.load(traj_path).astype(np.float32)
                goal_state_np = np.load(goal_path).astype(np.float32)

        except Exception as e:
            raise IOError(f"Failed to load .npy files for basename {basename} (idx {idx}): {e}") from e

        # Validate shapes and consistency
        if expert_trajectory_np.ndim != 2 or expert_trajectory_np.shape[0] == 0:
            raise ValueError(
                f"Expert trajectory for {basename} is malformed or empty. "
                f"Shape: {expert_trajectory_np.shape}, Expected: (L > 0, F > 0)"
            )
        if goal_state_np.ndim != 1:
            raise ValueError(f"Goal state for {basename} is malformed. Shape: {goal_state_np.shape}, Expected: (F > 0,)")
        if expert_trajectory_np.shape[1] != self.state_dim:
            raise ValueError(
                f"Feature dimension mismatch in trajectory for {basename}. "
                f"Expected {self.state_dim} (from first item), got {expert_trajectory_np.shape[1]}"
            )
        if goal_state_np.shape[0] != self.state_dim:
            raise ValueError(
                f"Feature dimension mismatch in goal for {basename}. "
                f"Expected {self.state_dim} (from first item), got {goal_state_np.shape[0]}"
            )

        initial_state_np = expert_trajectory_np[0, :]  # (F,)

        return {
            "initial_state": initial_state_np,
            "goal_state": goal_state_np,
            "expert_trajectory": expert_trajectory_np,
            "id": basename,
        }
