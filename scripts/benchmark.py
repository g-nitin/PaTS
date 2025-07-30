import argparse
import json
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from fastdtw import fastdtw  # type: ignore
from scipy.spatial.distance import hamming

from .BlocksWorldValidator import BlocksWorldValidator, ValidationResult
from .models.lstm import PaTS_LSTM
from .models.ttm import BlocksWorldTTM
from .models.ttm import ModelConfig as TTMModelConfig
from .models.xgboost import XGBoostPlanner

# Setup device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Benchmarking using device: {DEVICE}")


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

        # Check if the model was trained with the MLM task
        was_trained_with_mlm = checkpoint.get("use_mlm_task", False)

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
            use_mlm_task=was_trained_with_mlm,
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
    def __init__(self, model_path: Path, num_blocks: int, device: torch.device, encoding_type: str, seed: int):
        # XGBoost is CPU-bound, so device is not used but kept for interface consistency
        super().__init__(model_path, num_blocks, device)
        self.planner: XGBoostPlanner
        self.encoding_type = encoding_type
        self.seed = seed
        self._state_dim: Optional[int] = None

    def load_model(self):
        print(f"Loading XGBoost model from: {self.model_path}")
        self.planner = XGBoostPlanner.load(
            self.model_path, encoding_type=self.encoding_type, num_blocks=self.num_blocks, seed=self.seed
        )
        # Infer state_dim from the loaded model's structure
        if self.encoding_type == "bin":
            # Input to model is (current_state + goal_state), so divide by 2
            self._state_dim = self.planner.model.estimators_[0].n_features_in_ // 2  # type: ignore
        elif self.encoding_type == "sas":
            self._state_dim = self.num_blocks

        self.model = self.planner.model  # For consistency
        print(f"XGBoost model loaded. State dim inferred as: {self._state_dim}")

    def predict_sequence(self, initial_state_np: np.ndarray, goal_state_np: np.ndarray, max_length: int) -> List[List[int]]:
        if self.planner is None:
            raise RuntimeError("XGBoost model not loaded. Call load_model() first.")

        plan = [initial_state_np]
        current_state = initial_state_np

        for _ in range(max_length - 1):
            X_input = np.concatenate([current_state, goal_state_np]).reshape(1, -1)
            next_state = self.planner.predict(X_input).flatten()

            if np.array_equal(current_state, next_state):
                break

            plan.append(next_state)
            current_state = next_state

            if np.array_equal(current_state, goal_state_np):
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
            hamming_dist_raw = lambda u, v: hamming(u, v) * num_features

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
    model_type: str, model_path: Path, num_blocks: int, device: torch.device, encoding_type: str, seed: int
) -> PlannableModel:
    """Factory function to get a PlannableModel instance."""
    model_type_lower = model_type.lower()
    if model_type_lower == "ttm":
        return TTMWrapper(model_path, num_blocks, device)
    elif model_type_lower == "lstm":
        return LSTMWrapper(model_path, num_blocks, device)
    elif model_type_lower == "xgboost":
        return XGBoostWrapper(model_path, num_blocks, device, encoding_type, seed)
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
    data_root_dir = Path(args.dataset_dir)
    num_blocks = args.num_blocks
    model_type = args.model_type
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Construct paths based on num_blocks
    if not data_root_dir.exists():
        print(f"ERROR: Block-specific data directory not found: {data_root_dir}")
        return

    test_split_file = data_root_dir / "test_files.txt"
    problem_basenames = load_problem_basenames_from_split_file(test_split_file)
    if not problem_basenames:
        print(f"No test problems found in {test_split_file}. Exiting.")
        return

    print(f"Found {len(problem_basenames)} test problems for N={num_blocks}.")

    # Initialize Validator
    print(f"Initializing validator for N={num_blocks} with '{args.encoding_type}' encoding.")

    validator = None
    if args.encoding_type == "bin":
        # Construct the path to the predicate manifest for the specific num_blocks
        predicate_manifest_file = data_root_dir / f"predicate_manifest_{num_blocks}.txt"
        if not predicate_manifest_file.exists():
            print(f"ERROR: Predicate manifest file not found: {predicate_manifest_file}")
            return
        try:
            validator = BlocksWorldValidator(
                num_blocks, args.encoding_type, predicate_manifest_file=predicate_manifest_file
            )
            print(f"Validator initialized with state_size={validator.state_size}.")
        except Exception as e:
            print(f"ERROR: Failed to initialize BlocksWorldValidator for binary encoding: {e}")
            return
    elif args.encoding_type == "sas":
        try:
            # SAS encoding does not require a predicate manifest file.
            validator = BlocksWorldValidator(num_blocks, args.encoding_type)
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
            model_type, model_path, num_blocks, DEVICE, args.encoding_type, args.seed
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

    for i, basename in enumerate(problem_basenames):
        print(f"  Processing problem {i + 1}/{len(problem_basenames)}: {basename} ...", end="", flush=True)

        traj_bin_path = data_root_dir / "trajectories_bin" / f"{basename}.traj.{args.encoding_type}.npy"
        goal_bin_path = data_root_dir / "trajectories_bin" / f"{basename}.goal.{args.encoding_type}.npy"

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
        "--dataset_dir", type=str, required=True, help="Directory of the PaTS dataset (e.g., 'data/blocks_4')"
    )
    parser.add_argument("--num_blocks", type=int, required=True, help="Number of blocks for the problems to benchmark.")
    parser.add_argument(
        "--model_type", type=str, required=True, choices=["ttm", "lstm", "xgboost"], help="Type of model to benchmark."
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the trained model directory (TTM) or .pth file (LSTM)."
    )
    parser.add_argument(
        "--encoding_type",
        type=str,
        default="bin",
        choices=["bin", "sas"],
        help="The encoding type of the dataset to benchmark against.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./benchmark_outputs", help="Directory to save benchmark results."
    )
    parser.add_argument("--max_plan_length", type=int, default=50, help="Maximum plan length for model generation.")
    parser.add_argument("--save_detailed_results", action="store_true", help="Save detailed results for each problem.")
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed, mainly for XGBoost model initialization consistency. Default is 13.",
    )

    cli_args = parser.parse_args()
    run_benchmark(cli_args)
