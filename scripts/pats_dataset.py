from pathlib import Path
from typing import Any, Dict

import numpy as np
from torch.utils.data import Dataset


class PaTSDataset(Dataset):
    """
    A PyTorch Dataset for loading Planning as Time-Series (PaTS) data.
    It loads pre-encoded binary trajectories and goal states from .npy files.
    """

    def __init__(self, dataset_dir: str | Path, split_file_name: str, encoding_type: str = "binary"):
        """
        Initializes the PaTSDataset.

        :param dataset_dir: The root directory for a specific number of blocks (e.g., 'data/blocks_4/').
        :param split_file_name: The name of the file containing problem basenames for this split (e.g., 'train_files.txt').
        :param encoding_type: The encoding of the data to load ('binary' or 'sas').
        """
        self.dataset_dir = Path(dataset_dir)
        self.split_file_path = self.dataset_dir / split_file_name
        self.encoding_type = encoding_type

        self.trajectories_bin_dir = self.dataset_dir / "trajectories_bin"
        if not self.trajectories_bin_dir.is_dir():
            raise FileNotFoundError(f"Trajectories binary directory not found: {self.trajectories_bin_dir}")

        self.basenames = self._load_basenames()
        if not self.basenames:
            raise ValueError(
                f"No basenames loaded from {self.split_file_path}. "
                "Ensure the file exists, is not empty, and is in the correct directory."
            )

        self.state_dim = self._infer_state_dim()
        if self.state_dim <= 0:
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
        traj_path = self.trajectories_bin_dir / f"{first_basename}.traj.{self.encoding_type}.npy"
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

        traj_path = self.trajectories_bin_dir / f"{basename}.traj.{self.encoding_type}.npy"
        goal_path = self.trajectories_bin_dir / f"{basename}.goal.{self.encoding_type}.npy"

        if not traj_path.exists():
            raise FileNotFoundError(f"Trajectory file not found for basename {basename}: {traj_path}")
        if not goal_path.exists():
            raise FileNotFoundError(f"Goal file not found for basename {basename}: {goal_path}")

        try:
            # Load as float32 for direct use in PyTorch tensors
            expert_trajectory_np = np.load(traj_path).astype(np.float32)  # (L, F)
            goal_state_np = np.load(goal_path).astype(np.float32)  # (F,)
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
