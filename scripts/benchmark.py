import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from fastdtw import fastdtw  # type: ignore
from scipy.spatial.distance import hamming

from .BlocksWorldValidator import BlocksWorldValidator, ValidationResult
from .models.llama import LlamaWrapper
from .models.lstm import PaTS_LSTM
from .models.ttm import BlocksWorldTTM
from .models.ttm import ModelConfig as TTMModelConfig
from .models.xgboost import XGBoostPlanner
from .PlannableModel import PlannableModel

# Setup device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Benchmarking using device: {DEVICE}")


class TTMWrapper(PlannableModel):
    def __init__(self, model_path: Path, num_blocks: int, device: torch.device):
        super().__init__(model_path, num_blocks, device)
        self.ttm_instance: BlocksWorldTTM
        self.config: TTMModelConfig

    def load_model(self):
        print(f"Loading TTM model from: {self.model_path}")
        self.ttm_instance = BlocksWorldTTM.load(self.model_path, self.device)
        self.model = self.ttm_instance.model  # The actual torch model
        self.config = self.ttm_instance.config
        print(
            f"TTM model loaded. Context: {self.config.context_length}, Pred Len: {self.config.prediction_length}, State Dim: {self.config.state_dim}"
        )

    def predict_sequence(self, initial_state_np: np.ndarray, goal_state_np: np.ndarray, max_length: int) -> List[List[int]]:
        if self.model is None or self.config is None:
            raise RuntimeError("TTM model not loaded. Call load_model() first.")

        self.ttm_instance.model.eval()
        with torch.no_grad():
            # Convert numpy inputs to torch tensors and MOVE TO THE CORRECT DEVICE
            initial_state_tensor = torch.from_numpy(initial_state_np.astype(np.float32)).unsqueeze(0).to(self.device)
            goal_state_tensor = torch.from_numpy(goal_state_np.astype(np.float32)).unsqueeze(0).to(self.device)
            # For SAS+, inputs will be integer, so ensure they are passed as LongTensor before potentially being cast to float by TTM.predict
            if self.ttm_instance.config.encoding_type == "sas":
                initial_state_tensor = initial_state_tensor.long()
                goal_state_tensor = goal_state_tensor.long()

            # Initialize the context by repeating the initial state
            # This will now be on the correct device because initial_state_tensor is
            current_context = initial_state_tensor.unsqueeze(1).repeat(1, self.config.context_length, 1)

            # The plan starts with the initial state
            generated_plan_tensors = [initial_state_tensor.squeeze(0)]

            for step in range(max_length - 1):  # max_length includes S0
                # Predict the next sequence of states from the current context
                # The underlying predict method now takes the full context
                predicted_future_sequence = self.ttm_instance.predict(current_context, goal_state_tensor)  # (1, P, F)

                # We only need the very next state for our autoregressive step
                next_state = predicted_future_sequence[:, 0, :]  # (1, F)

                # Stagnation check: if the model predicts no change, stop.
                # This prevents infinitely long plans of repeating states.
                last_state_in_context = current_context[:, -1, :]
                if torch.equal(next_state, last_state_in_context.to(next_state.dtype)):  # Ensure type match for comparison
                    # print(f"TTM: Stagnation detected at step {step + 1}. Stopping.")
                    break

                generated_plan_tensors.append(next_state.squeeze(0))

                # Update the context for the next iteration:
                # Roll the context to the left and append the new state at the end
                current_context = torch.roll(current_context, shifts=-1, dims=1)
                current_context[:, -1, :] = next_state.to(current_context.dtype)  # Ensure type match for assignment

                # Goal achievement check
                if torch.equal(
                    next_state.squeeze(0), goal_state_tensor.squeeze(0).to(next_state.dtype)
                ):  # Ensure type match for comparison
                    # print(f"TTM: Goal reached at step {step + 1}.")
                    break

            # Convert the final list of tensors to the required List[List[int]] format
            final_plan_np = torch.stack(generated_plan_tensors).cpu().numpy()
            return [state.astype(int).tolist() for state in final_plan_np]

    @property
    def model_name(self) -> str:
        return f"TTM_{Path(self.config.ttm_model_path).name}" if self.config else "TTM_unloaded"

    @property
    def state_dim(self) -> int:
        if not self.config or self.config.state_dim is None:
            raise ValueError("TTM config not loaded or state_dim not set.")
        return self.config.state_dim


class LSTMWrapper(PlannableModel):
    def __init__(self, model_path: Path, num_blocks: int, device: torch.device):
        super().__init__(model_path, num_blocks, device)
        self.lstm_model: PaTS_LSTM
        self._state_dim: Optional[int] = None  # Will be loaded from checkpoint

    def load_model(self):
        if not self.model_path.is_file():
            raise FileNotFoundError(f"LSTM model file not found: {self.model_path}")

        print(f"Loading LSTM model from: {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Read model parameters from the checkpoint
        num_features = checkpoint.get("num_features")
        hidden_size = checkpoint.get("hidden_size")
        num_lstm_layers = checkpoint.get("num_lstm_layers")
        dropout_prob = checkpoint.get("dropout_prob", 0.2)
        encoding_type = checkpoint.get("encoding_type", "bin")  # Default to 'bin' if not found
        num_blocks = checkpoint.get("target_num_blocks")
        embedding_dim = checkpoint.get("embedding_dim")

        if not all([num_features, hidden_size, num_lstm_layers, num_blocks is not None]):
            raise ValueError(
                "LSTM checkpoint missing required parameters (num_features, hidden_size, num_lstm_layers, target_num_blocks)."
            )

        self._state_dim = num_features
        # Instantiate the model with the correct configuration based on the checkpoint
        self.lstm_model = PaTS_LSTM(
            num_features=num_features,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
            dropout_prob=dropout_prob,
            encoding_type=encoding_type,
            num_blocks=num_blocks,
            embedding_dim=embedding_dim,
        ).to(self.device)

        self.lstm_model.load_state_dict(checkpoint["model_state_dict"])
        self.lstm_model.eval()
        self.model = self.lstm_model  # For consistency with PlannableModel
        print(
            f"LSTM model loaded. Features: {num_features}, Hidden: {hidden_size}, "
            f"Layers: {num_lstm_layers}, Encoding: {encoding_type}"
        )

    def predict_sequence(self, initial_state_np: np.ndarray, goal_state_np: np.ndarray, max_length: int) -> List[List[int]]:
        if self.lstm_model is None:
            raise RuntimeError("LSTM model not loaded. Call load_model() first.")

        # generate_plan_lstm is a standalone function in lstm.py, let's adapt its core logic
        self.lstm_model.eval()
        with torch.no_grad():
            # Create tensors with the correct dtype based on the model's encoding type
            if self.lstm_model.encoding_type == "sas":
                # For SAS+, the input is integer indices and requires a LongTensor
                # Safeguard exists to clamp out-of-bounds indices from input files
                # This mirrors the safeguard in train_model.py and prevents CUDA asserts.
                # The valid indices are 0 (table) through N (block N), where N = num_blocks.
                max_valid_index = self.lstm_model.num_blocks
                initial_state_np.clip(min=0, max=max_valid_index, out=initial_state_np)
                goal_state_np.clip(min=0, max=max_valid_index, out=goal_state_np)
                current_S_tensor = torch.LongTensor(initial_state_np).unsqueeze(0).to(self.device)
                goal_S_tensor = torch.LongTensor(goal_state_np).unsqueeze(0).to(self.device)
            else:  # 'bin' encoding
                current_S_tensor = torch.FloatTensor(initial_state_np).unsqueeze(0).to(self.device)
                goal_S_tensor = torch.FloatTensor(goal_state_np).unsqueeze(0).to(self.device)

            h_prev = torch.zeros(self.lstm_model.num_lstm_layers, 1, self.lstm_model.hidden_size).to(self.device)
            c_prev = torch.zeros(self.lstm_model.num_lstm_layers, 1, self.lstm_model.hidden_size).to(self.device)

            generated_plan_tensors = [current_S_tensor.clone()]  # Start with S0

            for step in range(max_length - 1):  # Max_length includes S0
                next_S, _, h_next, c_next = self.lstm_model.predict_step(current_S_tensor, goal_S_tensor, h_prev, c_prev)
                generated_plan_tensors.append(next_S.clone())
                current_S_tensor = next_S
                h_prev, c_prev = h_next, c_next

                if torch.equal(current_S_tensor.squeeze(), goal_S_tensor.squeeze()):  # Compare 1D tensors
                    # print(f"LSTM: Goal reached at step {step + 1}.")
                    break

            final_plan_np = torch.cat(generated_plan_tensors, dim=0).cpu().numpy()
            return [state.astype(int).tolist() for state in final_plan_np]

    @property
    def model_name(self) -> str:
        return "PaTS_LSTM"

    @property
    def state_dim(self) -> int:
        if self._state_dim is None:
            raise ValueError("LSTM model not loaded or state_dim not set.")
        return self._state_dim


class XGBoostWrapper(PlannableModel):
    def __init__(
        self,
        model_path: Path,
        num_blocks: int,
        device: torch.device,
        encoding_type: str,
        seed: int,
        context_window_size: int = 1,
    ):
        # XGBoost is CPU-bound, so device is not used but kept for interface consistency
        super().__init__(model_path, num_blocks, device)
        self.planner: XGBoostPlanner
        self.encoding_type = encoding_type
        self.seed = seed
        self.context_window_size = context_window_size
        self._state_dim: Optional[int] = None
        self.loaded_context_window_size: int = (
            context_window_size  # This will be updated from the loaded model's config if available
        )

    def load_model(self):
        print(f"Loading XGBoost model from: {self.model_path}")
        self.planner = XGBoostPlanner.load(
            self.model_path, encoding_type=self.encoding_type, num_blocks=self.num_blocks, seed=self.seed
        )
        # Retrieve context_window_size from loaded model if it was saved with it
        if hasattr(self.planner, "context_window_size"):
            self.loaded_context_window_size = self.planner.context_window_size
        else:
            # Fallback for old models or if not saved. Warn user.
            print(
                f"Warning: 'context_window_size' not found in loaded XGBoost model. Using default: {self.loaded_context_window_size}"
            )

        # Infer state_dim from the loaded model's structure
        # Input to model is (context_window_size * state_dim + goal_state_dim)
        if self.encoding_type == "bin":  # For binary, state_dim is the number of predicates
            total_input_features = self.planner.model.estimators_[0].n_features_in_  # type: ignore
            self._state_dim = total_input_features // (self.loaded_context_window_size + 1)

        elif self.encoding_type == "sas":
            self._state_dim = self.num_blocks

        self.model = self.planner.model  # For consistency
        print(
            f"XGBoost model loaded. State dim inferred as: {self._state_dim}. Context window size: {self.loaded_context_window_size}"
        )

    def predict_sequence(self, initial_state_np: np.ndarray, goal_state_np: np.ndarray, max_length: int) -> List[List[int]]:
        if self.planner is None:
            raise RuntimeError("XGBoost model not loaded. Call load_model() first.")

        plan = [initial_state_np]

        # Maintain a deque or list for the sliding context window
        current_context_window = [initial_state_np] * self.loaded_context_window_size  # Initialize with S0 repeated

        for _ in range(max_length - 1):  # max_length includes S0
            # Construct the input for prediction
            # The context window should contain the last `loaded_context_window_size` states
            X_input_parts = current_context_window + [goal_state_np]
            X_input = np.concatenate(X_input_parts).reshape(1, -1)

            next_state = self.planner.predict(X_input).flatten()

            # Stagnation check: if the model predicts no change from the last state in the plan
            if np.array_equal(plan[-1], next_state):
                # print(f"XGBoost: Stagnation detected at step {len(plan)}. Stopping.")
                break

            plan.append(next_state)

            # Update the context window: remove oldest, add newest
            current_context_window.pop(0)
            current_context_window.append(next_state)

            # Goal achievement check
            if np.array_equal(next_state, goal_state_np):
                # print(f"XGBoost: Goal reached at step {len(plan) - 1}.")
                break

        return [state.tolist() for state in plan]  # type: ignore

    @property
    def model_name(self) -> str:
        return "PaTS_XGBoost"

    @property
    def state_dim(self) -> int:
        if self._state_dim is None:
            raise ValueError("XGBoost model not loaded or state_dim not inferred.")
        return self._state_dim


def compute_timeseries_metrics(
    predicted_plan: List[List[int]], expert_plan: np.ndarray, encoding_type: str
) -> Dict[str, float]:
    """
    Computes time-series specific metrics. Jaccard is only computed for 'bin' encoding.

    :param predicted_plan: The plan generated by the model, as a list of binary state lists.
    :param expert_plan: The expert plan from the dataset, as a NumPy array.
    :return: A dictionary containing the calculated metrics. Returns -1.0 for metrics that cannot be computed.
    """

    # Initial validation
    if not predicted_plan or expert_plan.shape[0] == 0:
        return {
            "mean_per_step_hamming_dist": -1.0,
            "mean_per_step_jaccard": -1.0,
            "dtw_distance": -1.0,
        }

    predicted_plan_np = np.array(predicted_plan, dtype=np.int8)

    # 1. Per-step metrics (Hamming, Jaccard)
    min_len = min(predicted_plan_np.shape[0], expert_plan.shape[0])
    step_hammings = []
    step_jaccards = []

    if min_len > 0:
        for i in range(min_len):
            pred_state = predicted_plan_np[i]
            expert_state = expert_plan[i]

            # Hamming Distance (raw count of differing bits)
            step_hammings.append(np.sum(pred_state != expert_state))

    mean_hamming = np.mean(step_hammings) if step_hammings else -1.0

    # 2. Dynamic Time Warping (DTW) using fastdtw
    # fastdtw is generally robust, but we avoid for empty sequences.
    if predicted_plan_np.shape[0] < 1 or expert_plan.shape[0] < 1:
        dtw_dist = -1.0
    else:
        try:
            # Use fastdtw and explicitly provide the distance function.
            # `scipy.spatial.distance.hamming` calculates the *proportion* of
            # differing elements. We multiply by the number of features to get the
            # raw count, which is a more intuitive DTW cost.
            num_features = predicted_plan_np.shape[1]

            # Define a lambda to wrap the hamming distance and scale it
            def hamming_dist_raw(u, v):
                return hamming(u, v) * num_features

            # The fastdtw function is the robust way to do this.
            dtw_dist, path = fastdtw(predicted_plan_np, expert_plan, dist=hamming_dist_raw)

        except Exception as e:
            print(f"\n[Warning] DTW calculation with fastdtw failed unexpectedly. Error: {e}")
            dtw_dist = -1.0

    mean_jaccard = -1.0
    if encoding_type == "bin":
        if min_len > 0:
            for i in range(min_len):
                pred_state = predicted_plan_np[i]
                expert_state = expert_plan[i]
                pred_true_indices = set(np.where(pred_state == 1)[0])
                expert_true_indices = set(np.where(expert_state == 1)[0])
                intersection_len = len(pred_true_indices.intersection(expert_true_indices))
                union_len = len(pred_true_indices.union(expert_true_indices))
                jaccard = intersection_len / union_len if union_len > 0 else 1.0
                step_jaccards.append(jaccard)
        mean_jaccard = np.mean(step_jaccards) if step_jaccards else -1.0

    return {
        "mean_per_step_hamming_dist": float(mean_hamming),
        "mean_per_step_jaccard": float(mean_jaccard),
        "dtw_distance": float(dtw_dist),
    }


def get_plannable_model(
    model_type: str,
    model_path: Path,
    num_blocks: int,
    device: torch.device,
    encoding_type: str,
    seed: int,
    context_window_size: int,
    dataset_dir: Path,
    llama_use_few_shot: bool = True,
) -> PlannableModel:
    """Factory function to get a PlannableModel instance."""
    model_type_lower = model_type.lower()
    if model_type_lower == "ttm":
        return TTMWrapper(model_path, num_blocks, device)
    elif model_type_lower == "lstm":
        return LSTMWrapper(model_path, num_blocks, device)
    elif model_type_lower == "xgboost":
        return XGBoostWrapper(model_path, num_blocks, device, encoding_type, seed, context_window_size)
    elif model_type_lower == "llama":
        # For Llama, model_path is the model_id string.
        # dataset_dir is needed by LlamaWrapper to fetch a one-shot example.
        return LlamaWrapper(
            str(model_path), num_blocks, device, encoding_type, dataset_dir, use_few_shot_example=llama_use_few_shot
        )
    # elif model_type_lower == "plansformer":
    #     # return PlansformerWrapper(model_path, num_blocks, device) # Placeholder
    #     raise NotImplementedError("Plansformer wrapper not yet implemented.")
    else:
        raise ValueError(f"Unsupported model type for PlannableModel factory: {model_type}")


# ** Helper Functions for Benchmarking **
def load_problem_basenames_from_split_file(split_file_path: Path) -> List[str]:
    if not split_file_path.exists():
        print(f"Warning: Split file not found: {split_file_path}")
        return []
    with open(split_file_path, "r") as f:
        basenames = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return basenames


def plot_benchmark_summary(
    aggregated_metrics: Dict[str, Any],
    model_type: str,
    encoding_type: str,
    num_blocks: int,
    output_dir: Path,
):
    """
    Plots key aggregated benchmark metrics as a bar chart.
    """
    sns.set_theme()
    metrics_to_plot = {
        "Solved Rate (Strict)": aggregated_metrics.get("solved_rate_strict", 0.0),
        "Valid Sequence Rate": aggregated_metrics.get("valid_sequence_rate", 0.0),
        "Goal Achievement Rate": aggregated_metrics.get("goal_achievement_rate", 0.0),
        "Avg Plan Length Ratio (Solved)": aggregated_metrics.get("avg_plan_length_ratio_for_solved", 0.0),
        "Avg DTW Distance": aggregated_metrics.get("avg_dtw_distance", 0.0),
        "Avg Hamming Distance": aggregated_metrics.get("avg_mean_per_step_hamming_dist", 0.0),
    }

    # Filter out metrics that are -1.0 (not computed or invalid)
    metrics_to_plot = {k: v for k, v in metrics_to_plot.items() if v >= 0}

    if not metrics_to_plot:
        print("No valid aggregated metrics to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.bar(
        metrics_to_plot.keys(), metrics_to_plot.values(), color=sns.color_palette("viridis", len(metrics_to_plot))
    )
    ax.set_ylabel("Value")
    ax.set_title(f"Benchmark Summary for {model_type} (N={num_blocks}, Encoding={encoding_type})")
    plt.xticks(rotation=45, ha="right")

    # Adjust ylim for DTW/Hamming if they are present and large
    max_val = max(metrics_to_plot.values()) if metrics_to_plot else 0.0
    plt.ylim(0, max(1.1, max_val * 1.1))  # Ensure y-axis starts from 0 and accommodates max value

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, round(yval, 2), ha="center", va="bottom")

    plt.tight_layout()
    plot_filename = f"benchmark_summary_{model_type}_N{num_blocks}_{encoding_type}.png"
    plt.savefig(output_dir / plot_filename)
    plt.close()
    print(f"Benchmark summary plot saved to {output_dir / plot_filename}")


def plot_violation_distribution(
    violation_counts: Dict[str, int],
    model_type: str,
    encoding_type: str,
    num_blocks: int,
    output_dir: Path,
):
    """
    Plots the distribution of violation codes as a bar chart.
    """
    if not violation_counts:
        print("No violation data to plot.")
        return

    sns.set_theme()
    sorted_violations = sorted(violation_counts.items(), key=lambda item: item[1], reverse=True)
    labels = [item[0] for item in sorted_violations]
    counts = [item[1] for item in sorted_violations]

    plt.figure(figsize=(12, 7))
    bars = sns.barplot(x=labels, y=counts, palette="rocket")
    plt.xlabel("Violation Code")
    plt.ylabel("Count")
    plt.title(f"Violation Distribution for {model_type} (N={num_blocks}, Encoding={encoding_type})")
    plt.xticks(rotation=45, ha="right")

    for bar in bars.patches:
        bars.annotate(
            f"{int(bar.get_height())}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            xytext=(0, 5),
            textcoords="offset points",
        )

    plt.tight_layout()
    plot_filename = f"violation_distribution_{model_type}_N{num_blocks}_{encoding_type}.png"
    plt.savefig(output_dir / plot_filename)
    plt.close()
    print(f"Violation distribution plot saved to {output_dir / plot_filename}")


def plot_plan_length_histograms(
    predicted_lengths: List[int],
    expert_lengths: List[int],
    model_type: str,
    encoding_type: str,
    num_blocks: int,
    output_dir: Path,
):
    """
    Plots histograms of predicted and expert plan lengths.
    """
    if not predicted_lengths and not expert_lengths:
        print("No plan length data to plot.")
        return

    sns.set_theme()
    plt.figure(figsize=(10, 6))

    bins = (
        range(min(predicted_lengths + expert_lengths), max(predicted_lengths + expert_lengths) + 2)
        if predicted_lengths or expert_lengths
        else range(1, 10)
    )

    if predicted_lengths:
        sns.histplot(predicted_lengths, bins=bins, kde=True, color="skyblue", label="Predicted Plan Lengths", alpha=0.6)
    if expert_lengths:
        sns.histplot(expert_lengths, bins=bins, kde=True, color="orange", label="Expert Plan Lengths", alpha=0.6)

    plt.xlabel("Plan Length")
    plt.ylabel("Frequency")
    plt.title(f"Plan Length Distribution for {model_type} (N={num_blocks}, Encoding={encoding_type})")
    plt.legend()
    plt.grid(axis="y", alpha=0.75)
    plt.tight_layout()
    plot_filename = f"plan_length_histograms_{model_type}_N{num_blocks}_{encoding_type}.png"
    plt.savefig(output_dir / plot_filename)
    plt.close()
    print(f"Plan length histograms saved to {output_dir / plot_filename}")


def plot_timeseries_metrics_histograms(
    dtw_distances: List[float],
    hamming_distances: List[float],
    model_type: str,
    encoding_type: str,
    num_blocks: int,
    output_dir: Path,
):
    """
    Plots histograms for DTW and Hamming distances.
    """
    if not dtw_distances and not hamming_distances:
        print("No time-series metrics data to plot.")
        return

    sns.set_theme()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Time-Series Metrics Distribution for {model_type} (N={num_blocks}, Encoding={encoding_type})")

    if dtw_distances:
        sns.histplot(dtw_distances, kde=True, color="green", ax=axes[0])
        axes[0].set_title("DTW Distance")
        axes[0].set_xlabel("Distance")
        axes[0].set_ylabel("Frequency")
        axes[0].grid(axis="y", alpha=0.75)

    if hamming_distances:
        sns.histplot(hamming_distances, kde=True, color="purple", ax=axes[1])
        axes[1].set_title("Mean Per-Step Hamming Distance")
        axes[1].set_xlabel("Distance")
        axes[1].set_ylabel("Frequency")
        axes[1].grid(axis="y", alpha=0.75)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent suptitle overlap
    plot_filename = f"timeseries_metrics_histograms_{model_type}_N{num_blocks}_{encoding_type}.png"
    plt.savefig(output_dir / plot_filename)
    plt.close()
    print(f"Time-series metrics histograms saved to {output_dir / plot_filename}")


def plot_trajectory_comparison(
    model_type: str,
    predicted_plan: List[List[int]],
    expert_plan: np.ndarray,
    problem_basename: str,
    encoding_type: str,
    num_blocks: int,
    output_dir: Path,
):
    """
    Visualizes the predicted vs. expert trajectory.
    For binary encoding, uses a heatmap. For SAS+, uses line plots.
    """
    sns.set_theme()
    pred_np = np.array(predicted_plan)
    expert_np = expert_plan

    max_len = max(pred_np.shape[0], expert_np.shape[0])
    num_features = pred_np.shape[1] if pred_np.shape[0] > 0 else expert_np.shape[1]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    fig.suptitle(f"Trajectory Comparison for {problem_basename} (N={num_blocks}, Encoding={encoding_type})", fontsize=16)

    if encoding_type == "bin":
        # Heatmap for binary encoding
        sns.heatmap(pred_np.T, cmap="Greys", cbar=False, ax=axes[0], yticklabels=False)
        axes[0].set_title("Predicted Trajectory (Binary)")
        axes[0].set_xlabel("Time Step")
        axes[0].set_ylabel("Feature Index")

        sns.heatmap(expert_np.T, cmap="Greys", cbar=False, ax=axes[1], yticklabels=False)
        axes[1].set_title("Expert Trajectory (Binary)")
        axes[1].set_xlabel("Time Step")
        axes[1].set_ylabel("Feature Index")  # This will be shared, but good to label

    elif encoding_type == "sas":
        # Line plot for SAS+ encoding
        # Each line represents a block's position over time
        for i in range(num_blocks):
            axes[0].plot(pred_np[:, i], label=f"b{i + 1}")
            axes[1].plot(expert_np[:, i], label=f"b{i + 1}")

        axes[0].set_title("Predicted Trajectory (SAS+)")
        axes[0].set_xlabel("Time Step")
        axes[0].set_ylabel("Block Position Value")
        axes[0].legend(loc="upper left", bbox_to_anchor=(1, 1))
        axes[0].set_ylim(-1.5, num_blocks + 0.5)  # SAS+ values are -1, 0, 1..N

        axes[1].set_title("Expert Trajectory (SAS+)")
        axes[1].set_xlabel("Time Step")
        axes[1].set_ylabel("Block Position Value")
        axes[1].legend(loc="upper left", bbox_to_anchor=(1, 1))
        axes[1].set_ylim(-1.5, num_blocks + 0.5)

    plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])  # Adjust for suptitle and legends
    plot_filename = f"trajectory_comparison_{model_type}_N{num_blocks}_{encoding_type}_{problem_basename}.png"
    plot_path = output_dir / "trajectory_plots" / plot_filename
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
    # print(f"Trajectory comparison plot saved to {plot_path}") # Too verbose for every problem


def compute_aggregated_metrics(
    results: List[ValidationResult], expert_plan_lengths: List[int], timeseries_metrics_list: List[Dict[str, float]]
) -> Dict[str, Any]:
    num_samples = len(results)
    if num_samples == 0:
        return {"message": "No results to aggregate."}

    agg_metrics = {
        "num_samples": num_samples,
        "valid_sequence_rate": 0.0,  # Plan is fully valid by validator rules
        "goal_achievement_rate": 0.0,  # Final state is goal state
        "solved_rate_strict": 0.0,  # Valid sequence AND final state is goal
        "avg_predicted_plan_length": 0.0,
        "avg_expert_plan_length": np.mean(expert_plan_lengths) if expert_plan_lengths else 0.0,
        "avg_goal_jaccard_score": 0.0,
        "avg_goal_f1_score": 0.0,
        "avg_percent_physically_valid_states": 0.0,
        "avg_percent_valid_transitions": 0.0,
        "avg_plan_length_ratio_for_solved": 0.0,  # For strictly solved problems
        "avg_plan_length_diff_for_solved": 0.0,  # For strictly solved problems
        "avg_mean_per_step_hamming_dist": 0.0,
        "avg_mean_per_step_jaccard": 0.0,
        "avg_dtw_distance": 0.0,
        "violation_code_counts": Counter(),
    }

    valid_sequences_count = 0
    goal_achieved_count = 0
    solved_strict_count = 0
    total_predicted_plan_length = 0
    total_jaccard = 0.0
    total_f1 = 0.0
    total_perc_phys_valid_states = 0.0
    total_perc_valid_transitions = 0.0

    solved_plan_length_ratios = []
    solved_plan_length_diffs = []

    total_hamming = 0.0
    total_ts_jaccard = 0.0
    total_dtw = 0.0
    valid_ts_metrics_count = 0

    for ts_metrics in timeseries_metrics_list:
        if ts_metrics and ts_metrics.get("dtw_distance", -1.0) >= 0:
            total_hamming += ts_metrics["mean_per_step_hamming_dist"]
            total_ts_jaccard += ts_metrics["mean_per_step_jaccard"]
            total_dtw += ts_metrics["dtw_distance"]
            valid_ts_metrics_count += 1

    for i, res in enumerate(results):
        if res.is_valid:
            valid_sequences_count += 1
        if res.metrics.get("goal_achievement", 0.0) == 1.0:
            goal_achieved_count += 1

        is_strictly_solved = res.is_valid and res.metrics.get("goal_achievement", 0.0) == 1.0
        if is_strictly_solved:
            solved_strict_count += 1
            if expert_plan_lengths and i < len(expert_plan_lengths) and expert_plan_lengths[i] > 0:
                ratio = res.predicted_plan_length / expert_plan_lengths[i]
                diff = res.predicted_plan_length - expert_plan_lengths[i]
                solved_plan_length_ratios.append(ratio)
                solved_plan_length_diffs.append(diff)

        total_predicted_plan_length += res.predicted_plan_length
        total_jaccard += res.goal_jaccard_score
        total_f1 += res.goal_f1_score
        total_perc_phys_valid_states += res.metrics.get("percent_physically_valid_states", 0.0)
        total_perc_valid_transitions += res.metrics.get("percent_valid_transitions", 0.0)

        for violation in res.violations:
            agg_metrics["violation_code_counts"][violation.code] += 1

    agg_metrics["valid_sequence_rate"] = valid_sequences_count / num_samples
    agg_metrics["goal_achievement_rate"] = goal_achieved_count / num_samples
    agg_metrics["solved_rate_strict"] = solved_strict_count / num_samples
    agg_metrics["avg_predicted_plan_length"] = total_predicted_plan_length / num_samples
    agg_metrics["avg_goal_jaccard_score"] = total_jaccard / num_samples
    agg_metrics["avg_goal_f1_score"] = total_f1 / num_samples
    agg_metrics["avg_percent_physically_valid_states"] = total_perc_phys_valid_states / num_samples
    agg_metrics["avg_percent_valid_transitions"] = total_perc_valid_transitions / num_samples

    if valid_ts_metrics_count > 0:
        agg_metrics["avg_mean_per_step_hamming_dist"] = total_hamming / valid_ts_metrics_count
        agg_metrics["avg_mean_per_step_jaccard"] = total_ts_jaccard / valid_ts_metrics_count
        agg_metrics["avg_dtw_distance"] = total_dtw / valid_ts_metrics_count

    if solved_plan_length_ratios:
        agg_metrics["avg_plan_length_ratio_for_solved"] = np.mean(solved_plan_length_ratios)
        agg_metrics["avg_plan_length_diff_for_solved"] = np.mean(solved_plan_length_diffs)

    # Convert Counter to dict for JSON serialization
    agg_metrics["violation_code_counts"] = dict(agg_metrics["violation_code_counts"])

    return agg_metrics


# ** Main Benchmarking Logic **
def run_benchmark(args: argparse.Namespace):
    raw_block_dir = Path(args.dataset_dir)
    processed_block_encoding_dir = Path(args.processed_block_encoding_dir)
    num_blocks = args.num_blocks
    model_type = args.model_type
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Construct paths based on num_blocks
    if not raw_block_dir.exists():
        print(f"ERROR: Raw problem data directory not found: {raw_block_dir}")
        return

    test_split_file = raw_block_dir / "splits" / "test_files.txt"  # Split files are now in a 'splits' subdir

    problem_basenames = load_problem_basenames_from_split_file(test_split_file)
    if not problem_basenames:
        print(f"No test problems found in {test_split_file}. Exiting.")
        return

    print(f"Found {len(problem_basenames)} test problems for N={num_blocks}.")

    # problem_basenames = problem_basenames[:1]
    # print(f"~~Cutting length to {len(problem_basenames)}~~")

    # Initialize Validator
    print(f"Initializing validator for N={num_blocks} with '{args.encoding_type}' encoding.")

    validator = None
    if args.encoding_type == "bin":
        # Construct the path to the predicate manifest for the specific num_blocks
        try:
            validator = BlocksWorldValidator(num_blocks, args.encoding_type, raw_data_dir=raw_block_dir)
            print(f"Validator initialized with state_size={validator.state_size}.")
        except Exception as e:
            print(f"ERROR: Predicate manifest not found at {raw_block_dir / f'predicate_manifest_{num_blocks}.txt'}.")
            print("Please ensure the manifest file exists in the raw_data_dir.")
            print(f"ERROR: Failed to initialize BlocksWorldValidator for binary encoding: {e}")
            return
    elif args.encoding_type == "sas":
        try:
            # SAS encoding does not require a predicate manifest file.
            # But the validator constructor still expects raw_data_dir
            validator = BlocksWorldValidator(num_blocks, args.encoding_type, raw_data_dir=raw_block_dir)
            print(f"Validator initialized with state_size={validator.state_size}.")
        except Exception as e:
            print(f"ERROR: Failed to initialize BlocksWorldValidator for SAS encoding: {e}")
            return
    else:
        print(f"ERROR: Unknown encoding type '{args.encoding_type}'")
        return
    print(f"Validator initialized with state_size={validator.state_size}.")

    # Initialize Model
    try:
        wrapped_model: PlannableModel = get_plannable_model(
            model_type,
            model_path,
            num_blocks,
            DEVICE,
            args.encoding_type,
            args.seed,
            context_window_size=args.xgboost_context_window_size,
            dataset_dir=raw_block_dir,
            llama_use_few_shot=args.llama_use_few_shot,
        )
    except ValueError as e:
        print(f"ERROR: Could not initialize model: {e}")
        return  # or exit
    except NotImplementedError as e:
        print(f"ERROR: Model type not implemented: {e}")
        return  # or exit

    wrapped_model.load_model()

    # Verify state_dim consistency
    if wrapped_model.state_dim != validator.state_size:
        print("CRITICAL ERROR: State dimension mismatch!")
        print(f"  Model ({wrapped_model.model_name}) expects/produces state_dim = {wrapped_model.state_dim}")
        print(f"  Validator (from manifest/config) expects state_size = {validator.state_size}")
        print("  Ensure the model was trained with data generated using this manifest or an equivalent encoding.")
        return

    print(f"Benchmarking model: {wrapped_model.model_name}")

    all_validation_results: List[ValidationResult] = []
    all_expert_plan_lengths: List[int] = []
    all_timeseries_metrics: List[Dict[str, float]] = []

    # Max plan length for generation (can be different from TTM's fixed prediction length)
    # For LSTM, this limits the generation loop. For TTM, its output is fixed length.
    # The validator will check if goal is met within the *generated* plan.
    max_generation_steps = args.max_plan_length

    all_validation_results: List[ValidationResult] = []
    all_expert_plan_lengths: List[int] = []
    all_timeseries_metrics: List[Dict[str, float]] = []

    # New lists for plotting histograms
    all_predicted_plan_lengths: List[int] = []
    all_dtw_distances: List[float] = []
    all_hamming_distances: List[float] = []

    # Counter for detailed trajectory plots
    solved_problems_plotted_count = 0
    MAX_TRAJECTORY_PLOTS = 5  # Limit the number of individual trajectory plots

    for i, basename in enumerate(problem_basenames):
        print(f"  Processing problem {i + 1}/{len(problem_basenames)}: {basename} ...", end="", flush=True)

        traj_bin_path = processed_block_encoding_dir / f"{basename}.traj.{args.encoding_type}.npy"
        goal_bin_path = processed_block_encoding_dir / f"{basename}.goal.{args.encoding_type}.npy"

        if not traj_bin_path.exists() or not goal_bin_path.exists():
            print(" skipped (data files missing).")
            continue

        try:
            expert_trajectory_np = np.load(traj_bin_path)  # (L_expert, F)
            goal_state_np = np.load(goal_bin_path)  # (F,)
        except Exception as e:
            print(f" skipped (error loading npy: {e}).")
            continue

        if expert_trajectory_np.ndim != 2 or expert_trajectory_np.shape[0] == 0:
            print(f" skipped (expert trajectory malformed: shape {expert_trajectory_np.shape}).")
            continue

        initial_state_np = expert_trajectory_np[0]  # S0 (F,)
        all_expert_plan_lengths.append(expert_trajectory_np.shape[0])

        # Predict plan
        predicted_plan_list_of_lists = wrapped_model.predict_sequence(initial_state_np, goal_state_np, max_generation_steps)

        # Validate
        validation_res = validator.validate_sequence(predicted_plan_list_of_lists, goal_state_np.tolist())  # type: ignore
        all_validation_results.append(validation_res)

        # Compute and store time-series metrics
        ts_metrics = compute_timeseries_metrics(predicted_plan_list_of_lists, expert_trajectory_np, args.encoding_type)
        all_timeseries_metrics.append(ts_metrics)

        all_predicted_plan_lengths.append(validation_res.predicted_plan_length)
        if ts_metrics.get("dtw_distance", -1.0) >= 0:
            all_dtw_distances.append(ts_metrics["dtw_distance"])
        if ts_metrics.get("mean_per_step_hamming_dist", -1.0) >= 0:
            all_hamming_distances.append(ts_metrics["mean_per_step_hamming_dist"])

        # Plot individual trajectory comparison for solved problems
        if args.save_detailed_results and validation_res.is_valid and validation_res.metrics.get("goal_achievement") == 1.0:
            if solved_problems_plotted_count < MAX_TRAJECTORY_PLOTS:
                plot_trajectory_comparison(
                    model_type,
                    predicted_plan_list_of_lists,
                    expert_trajectory_np,
                    basename,
                    args.encoding_type,
                    args.num_blocks,
                    output_dir,
                )
                solved_problems_plotted_count += 1

        status = "VALID" if validation_res.is_valid else "INVALID"
        goal_reached = "GOAL" if validation_res.metrics.get("goal_achievement") == 1.0 else "NO_GOAL"
        print(
            f" done. Status: {status}, {goal_reached}. Len: {validation_res.predicted_plan_length}. DTW: {ts_metrics['dtw_distance']:.2f}"
        )

    # Aggregate and save results
    print("\nAggregating results...")
    aggregated_metrics = compute_aggregated_metrics(all_validation_results, all_expert_plan_lengths, all_timeseries_metrics)

    print("\n** Aggregated Benchmark Metrics **")
    for key, value in aggregated_metrics.items():
        if isinstance(value, dict) or isinstance(value, Counter):  # For violation_code_counts
            print(f"  {key}:")
            for k_inner, v_inner in value.items():
                print(f"    {k_inner}: {v_inner}")
        elif isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    results_filename = f"benchmark_results_{model_type}_{Path(model_path).stem}_N{num_blocks}.json"
    results_filepath = output_dir / results_filename
    with open(results_filepath, "w") as f:
        # Convert Counter to dict for JSON
        if "violation_code_counts" in aggregated_metrics and isinstance(
            aggregated_metrics["violation_code_counts"], Counter
        ):
            aggregated_metrics["violation_code_counts"] = dict(aggregated_metrics["violation_code_counts"])
        json.dump(aggregated_metrics, f, indent=4)
    print(f"\nBenchmark results saved to: {results_filepath}")

    # Generate plots for aggregated results
    print("\nGenerating benchmark plots...")
    plot_benchmark_summary(
        aggregated_metrics,
        model_type,
        args.encoding_type,
        num_blocks,
        output_dir,
    )
    plot_violation_distribution(
        aggregated_metrics["violation_code_counts"],  # This is already a dict
        model_type,
        args.encoding_type,
        num_blocks,
        output_dir,
    )
    plot_plan_length_histograms(
        all_predicted_plan_lengths,
        all_expert_plan_lengths,
        model_type,
        args.encoding_type,
        num_blocks,
        output_dir,
    )
    plot_timeseries_metrics_histograms(
        all_dtw_distances,
        all_hamming_distances,
        model_type,
        args.encoding_type,
        num_blocks,
        output_dir,
    )
    print("All benchmark plots generated.")

    # Optionally save detailed per-problem results
    if args.save_detailed_results:
        detailed_results_list = []
        for i, res in enumerate(all_validation_results):
            # Get the corresponding time-series metrics
            ts_metrics_for_problem = all_timeseries_metrics[i] if i < len(all_timeseries_metrics) else {}

            detailed_entry = {
                "problem_basename": problem_basenames[i] if i < len(problem_basenames) else "N/A",
                "is_valid": res.is_valid,
                "predicted_plan_length": res.predicted_plan_length,
                "expert_plan_length": all_expert_plan_lengths[i] if i < len(all_expert_plan_lengths) else -1,
                "goal_achievement": res.metrics.get("goal_achievement", 0.0),
                "goal_jaccard_score": res.goal_jaccard_score,
                "goal_f1_score": res.goal_f1_score,
                "timeseries_metrics": ts_metrics_for_problem,
                "violations": [{"code": v.code, "message": v.message, "details": v.details} for v in res.violations],
            }
            detailed_results_list.append(detailed_entry)

        detailed_filename = f"detailed_benchmark_results_{model_type}_{Path(model_path).stem}_N{num_blocks}.json"
        detailed_filepath = output_dir / detailed_filename
        with open(detailed_filepath, "w") as f_detailed:
            json.dump(detailed_results_list, f_detailed, indent=2)
        print(f"Detailed per-problem results saved to: {detailed_filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PaTS Model Benchmarking Script")
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        required=True,
        help="Path to the raw problem data directory for a specific N (e.g., 'data/raw_problems/blocksworld/N4')",
    )
    parser.add_argument(
        "--processed_block_encoding_dir",
        type=Path,
        required=True,
        help="Path to the processed block encoding directory (e.g., 'data/processed_trajectories/blocksworld/N4/bin')",
    )
    parser.add_argument(
        "--num_blocks",
        type=int,
        required=True,
        help="Number of blocks for the problems to benchmark.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["ttm", "lstm", "xgboost", "llama"],
        help="Type of model to benchmark.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model directory (TTM) or .pth file (LSTM).",
    )
    parser.add_argument(
        "--encoding_type",
        type=str,
        default="bin",
        choices=["bin", "sas"],
        help="The encoding type of the dataset to benchmark against.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./benchmark_outputs",
        help="Directory to save benchmark results.",
    )
    parser.add_argument("--max_plan_length", type=int, default=50, help="Maximum plan length for model generation.")
    parser.add_argument("--save_detailed_results", action="store_true", help="Save detailed results for each problem.")
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed, mainly for XGBoost model initialization consistency. Default is 13.",
    )
    parser.add_argument(
        "--xgboost_context_window_size",
        type=int,
        default=1,
        help="Number of past states to include in XGBoost input features. Must match training.",
    )
    parser.add_argument(
        "--llama_use_few_shot",
        action="store_true",
        help="For Llama models, include a one-shot example in the prompt for few-shot inference.",
    )

    cli_args = parser.parse_args()
    run_benchmark(cli_args)
