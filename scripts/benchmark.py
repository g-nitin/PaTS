import argparse
import json
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from BlocksWorldValidator import BlocksWorldValidator, ValidationResult
from models.lstm import PaTS_LSTM
from models.ttm import BlocksWorldTTM
from models.ttm import ModelConfig as TTMModelConfig

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


# ** TTM Wrapper **
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

        # TTM's predict method handles batching, but here we do one at a time for simplicity in benchmark loop
        # Inputs to TTM's predict are 0/1 torch tensors
        initial_state_tensor = torch.from_numpy(initial_state_np.astype(np.float32)).unsqueeze(0)  # (1, F)
        goal_state_tensor = torch.from_numpy(goal_state_np.astype(np.float32)).unsqueeze(0)  # (1, F)

        # The TTM's internal predict method uses its configured prediction_length.
        # We might want to allow overriding max_length if TTM supports variable length generation,
        # or if we need to truncate/extend its fixed-length output.
        # For now, we rely on TTM's prediction_length.
        # The `max_length` argument here is more of a general guideline for the benchmark.

        # The existing TTM predict method returns a tensor of shape (batch_size, prediction_length, num_features)
        # with values 0.0 or 1.0.
        predicted_plan_tensor_01 = self.ttm_instance.predict(initial_state_tensor, goal_state_tensor)  # (1, pred_len, F)

        # Convert to List[List[int]]
        # Squeeze batch dimension, convert to numpy, then to list of lists
        predicted_plan_np_01 = predicted_plan_tensor_01.squeeze(0).cpu().numpy()

        # The TTM output is the sequence of *next* states. We need to prepend the initial state.
        # However, TTM's `predict` is designed to forecast `prediction_length` steps *from* the context.
        # The output `predictions_original_binary` in TTM's `predict` is already the sequence of predicted future states.
        # We need to decide if the "plan" includes S0. Classical plans usually start from S0.
        # If TTM output is S1...ST, we prepend S0.
        # Let's assume TTM output is S1...Spred_len.

        # The current TTM `predict` method in `ttm.py` returns the *predicted future states*.
        # So, we should prepend the initial state to form the full trajectory.
        full_trajectory_np = np.vstack((initial_state_np, predicted_plan_np_01))

        # Truncate to max_length if necessary (though TTM has fixed prediction_length)
        # This max_length is for the benchmark, not for TTM's internal generation limit.
        # The validator will check if goal is reached within this.
        if full_trajectory_np.shape[0] > max_length:
            # This truncation might be aggressive if TTM needs its full prediction length to reach goal
            # print(f"Warning: TTM prediction truncated from {full_trajectory_np.shape[0]} to {max_length}")
            pass  # For now, let TTM produce its sequence, validator will check goal achievement

        return [state.astype(int).tolist() for state in full_trajectory_np]

    @property
    def model_name(self) -> str:
        return f"TTM_{Path(self.config.ttm_model_path).name}" if self.config else "TTM_unloaded"

    @property
    def state_dim(self) -> int:
        if not self.config or self.config.state_dim is None:
            raise ValueError("TTM config not loaded or state_dim not set.")
        return self.config.state_dim


# ** LSTM Wrapper **
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

        num_features = checkpoint.get("num_features")
        hidden_size = checkpoint.get("hidden_size")
        num_lstm_layers = checkpoint.get("num_lstm_layers")
        # dropout_prob is not typically saved, assume a default or make it part of config if needed

        if not all([num_features, hidden_size, num_lstm_layers]):
            raise ValueError("LSTM checkpoint missing required parameters (num_features, hidden_size, num_lstm_layers).")

        self._state_dim = num_features
        self.lstm_model = PaTS_LSTM(num_features, hidden_size, num_lstm_layers).to(self.device)
        self.lstm_model.load_state_dict(checkpoint["model_state_dict"])
        self.lstm_model.eval()
        self.model = self.lstm_model  # For consistency with PlannableModel
        print(f"LSTM model loaded. Features: {num_features}, Hidden: {hidden_size}, Layers: {num_lstm_layers}")

    def predict_sequence(self, initial_state_np: np.ndarray, goal_state_np: np.ndarray, max_length: int) -> List[List[int]]:
        if self.lstm_model is None:
            raise RuntimeError("LSTM model not loaded. Call load_model() first.")

        # generate_plan_lstm is a standalone function in lstm.py, let's adapt its core logic
        self.lstm_model.eval()
        with torch.no_grad():
            current_S_tensor = torch.FloatTensor(initial_state_np).unsqueeze(0).to(self.device)  # (1, F)
            goal_S_tensor = torch.FloatTensor(goal_state_np).unsqueeze(0).to(self.device)  # (1, F)

            h_prev = torch.zeros(self.lstm_model.num_lstm_layers, 1, self.lstm_model.hidden_size).to(self.device)
            c_prev = torch.zeros(self.lstm_model.num_lstm_layers, 1, self.lstm_model.hidden_size).to(self.device)

            generated_plan_tensors = [current_S_tensor.clone()]  # Start with S0

            for step in range(max_length - 1):  # Max_length includes S0
                next_S_binary, _, h_next, c_next = self.lstm_model.predict_step(
                    current_S_tensor, goal_S_tensor, h_prev, c_prev
                )
                generated_plan_tensors.append(next_S_binary.clone())
                current_S_tensor = next_S_binary
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


def get_plannable_model(model_type: str, model_path: Path, num_blocks: int, device: torch.device) -> PlannableModel:
    """Factory function to get a PlannableModel instance."""
    model_type_lower = model_type.lower()
    if model_type_lower == "ttm":
        return TTMWrapper(model_path, num_blocks, device)
    elif model_type_lower == "lstm":
        return LSTMWrapper(model_path, num_blocks, device)
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


def compute_aggregated_metrics(results: List[ValidationResult], expert_plan_lengths: List[int]) -> Dict[str, Any]:
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
    block_specific_data_dir = data_root_dir / f"blocks_{num_blocks}"
    if not block_specific_data_dir.exists():
        print(f"ERROR: Block-specific data directory not found: {block_specific_data_dir}")
        return

    predicate_manifest_file = block_specific_data_dir / f"predicate_manifest_{num_blocks}.txt"
    if not predicate_manifest_file.exists():
        print(f"ERROR: Predicate manifest file not found: {predicate_manifest_file}")
        return

    test_split_file = block_specific_data_dir / "test_files.txt"
    problem_basenames = load_problem_basenames_from_split_file(test_split_file)
    if not problem_basenames:
        print(f"No test problems found in {test_split_file}. Exiting.")
        return

    print(f"Found {len(problem_basenames)} test problems for N={num_blocks}.")

    # Initialize Validator
    validator = BlocksWorldValidator(num_blocks, predicate_manifest_file)
    print(f"Validator initialized for N={num_blocks} with state_size={validator.state_size}.")

    # Initialize Model
    try:
        wrapped_model: PlannableModel = get_plannable_model(model_type, model_path, num_blocks, DEVICE)
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
        print(f"  Validator (from manifest {predicate_manifest_file}) expects state_size = {validator.state_size}")
        print("  Ensure the model was trained with data generated using this manifest or an equivalent encoding.")
        return

    print(f"Benchmarking model: {wrapped_model.model_name}")

    all_validation_results: List[ValidationResult] = []
    all_expert_plan_lengths: List[int] = []

    # Max plan length for generation (can be different from TTM's fixed prediction length)
    # For LSTM, this limits the generation loop. For TTM, its output is fixed length.
    # The validator will check if goal is met within the *generated* plan.
    max_generation_steps = args.max_plan_length

    for i, basename in enumerate(problem_basenames):
        print(f"  Processing problem {i + 1}/{len(problem_basenames)}: {basename} ...", end="", flush=True)

        traj_bin_path = block_specific_data_dir / "trajectories_bin" / f"{basename}.traj.bin.npy"
        goal_bin_path = block_specific_data_dir / "trajectories_bin" / f"{basename}.goal.bin.npy"

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
        validation_res = validator.validate_sequence(predicted_plan_list_of_lists, goal_state_np)  # type: ignore
        all_validation_results.append(validation_res)

        status = "VALID" if validation_res.is_valid else "INVALID"
        goal_reached = "GOAL" if validation_res.metrics.get("goal_achievement") == 1.0 else "NO_GOAL"
        print(
            f" done. Status: {status}, {goal_reached}. Length: {validation_res.predicted_plan_length}. Violations: {len(validation_res.violations)}"
        )

    # Aggregate and save results
    print("\nAggregating results...")
    aggregated_metrics = compute_aggregated_metrics(all_validation_results, all_expert_plan_lengths)

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
            detailed_results_list.append(
                {
                    "problem_basename": problem_basenames[i] if i < len(problem_basenames) else "N/A",
                    "is_valid": res.is_valid,
                    "predicted_plan_length": res.predicted_plan_length,
                    "expert_plan_length": all_expert_plan_lengths[i] if i < len(all_expert_plan_lengths) else -1,
                    "goal_achievement": res.metrics.get("goal_achievement", 0.0),
                    "goal_jaccard_score": res.goal_jaccard_score,
                    "goal_f1_score": res.goal_f1_score,
                    "violations": [{"code": v.code, "message": v.message, "details": v.details} for v in res.violations],
                }
            )
        detailed_filename = f"detailed_benchmark_results_{model_type}_{Path(model_path).stem}_N{num_blocks}.json"
        detailed_filepath = output_dir / detailed_filename
        with open(detailed_filepath, "w") as f_detailed:
            json.dump(detailed_results_list, f_detailed, indent=2)
        print(f"Detailed per-problem results saved to: {detailed_filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PaTS Model Benchmarking Script")
    parser.add_argument(
        "--dataset_dir", type=str, required=True, help="Root directory of the PaTS dataset (e.g., './data')"
    )
    parser.add_argument("--num_blocks", type=int, required=True, help="Number of blocks for the problems to benchmark.")
    parser.add_argument(
        "--model_type", type=str, required=True, choices=["ttm", "lstm"], help="Type of model to benchmark."
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the trained model directory (TTM) or .pth file (LSTM)."
    )
    parser.add_argument(
        "--output_dir", type=str, default="./benchmark_outputs", help="Directory to save benchmark results."
    )
    parser.add_argument("--max_plan_length", type=int, default=50, help="Maximum plan length for model generation.")
    parser.add_argument("--save_detailed_results", action="store_true", help="Save detailed results for each problem.")

    cli_args = parser.parse_args()
    run_benchmark(cli_args)
