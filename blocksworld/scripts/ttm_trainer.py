import argparse
import json
import math
import posixpath
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path, PosixPath
from pprint import pformat
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, random_split
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer import Trainer
from transformers.trainer_callback import EarlyStoppingCallback
from transformers.trainer_utils import set_seed
from transformers.training_args import TrainingArguments
from tsfm_public import TrackingCallback
from tsfm_public.toolkit.get_model import get_model

# ** Constants **
DEFAULT_TTM_MODEL_PATH = "ibm-granite/granite-timeseries-ttm-r2"  # Default TTM model
SUPPORTED_CONTEXT_LENGTHS = [52, 90, 180, 360, 520, 1024, 1536]
SUPPORTED_PREDICTION_LENGTHS = [16, 30, 48, 60, 96, 192, 336, 720]

global DEVICE
DEVICE: torch.device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
# DEVICE = torch.device("cpu")
logger.info(f"Using device: {DEVICE}")


# ** Data Classes **
@dataclass
class ModelConfig:
    context_length: int
    prediction_length: int
    learning_rate: float
    batch_size: int
    num_epochs: int
    ttm_model_path: str
    seed: int
    state_dim: Optional[int] = None  # Will be set during training or loaded


@dataclass
class BlocksWorldSample:
    initial_state: List[int]
    goal_state: List[int]
    plan: List[List[int]]  # Sequence of state vectors
    actions: List[List[str]]  # Sequence of action descriptions
    feature_names: List[str]


# ** Dataset Class **
class BlocksWorldDataset(Dataset):
    def __init__(self, data_path: str, context_length: int, prediction_length: int):
        self.context_length: int = context_length
        self.prediction_length: int = prediction_length

        logger.info(f"Loading dataset from: {data_path}")
        with open(data_path, "r") as f:
            raw_data = json.load(f)["plans"]

        self.samples: List[BlocksWorldSample] = []
        for item in raw_data:
            sample = BlocksWorldSample(
                initial_state=item["initial_state"],
                goal_state=item["goal_state"],
                plan=item["plan"],
                actions=item["actions"],
                feature_names=item["feature_names"],
            )
            self.samples.append(sample)

        if not self.samples:
            raise ValueError(f"No samples found in dataset: {data_path}")

        # Get dimensionality from first sample
        self.state_dim: int = len(self.samples[0].initial_state)
        logger.info(f"Dataset loaded: {len(self.samples)} samples, state_dim={self.state_dim}")

        # Log a note about future_observed_mask padding strategy
        logger.debug(
            "Future values padding: Using mask=1.0 for goal_state padding in future_values. "
            "Loss WILL BE COMPUTED for these padded steps, enforcing goal prediction. "
            "This change was made to avoid issues with empty tensors on MPS backend when the mask might otherwise be all zeros."
        )

    def __len__(self):  # Length of the Dataset
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        plan_states_np = np.array(sample.plan, dtype=np.float32)  # shape (plan_len, state_dim)
        goal_state_np = np.array(sample.goal_state, dtype=np.float32)  # shape (state_dim,)
        plan_len = len(sample.plan)
        initial_state_np = plan_states_np[0]  # Assuming plan always includes initial state

        # Past values and mask
        past_values_np = np.zeros((self.context_length, self.state_dim), dtype=np.float32)
        past_observed_mask_np = np.zeros((self.context_length, self.state_dim), dtype=np.float32)

        # Number of actual plan steps to copy into the context window
        num_plan_steps_for_context = 0
        if plan_len > 0:  # Only proceed if there are plan steps
            num_plan_steps_for_context = min(plan_len, self.context_length)

        if num_plan_steps_for_context > 0:
            # Copy the actual plan steps
            past_values_np[:num_plan_steps_for_context] = plan_states_np[:num_plan_steps_for_context]
            past_observed_mask_np[:num_plan_steps_for_context, :] = 1.0

            # If padding is needed for the rest of the context window
            if num_plan_steps_for_context < self.context_length:
                # Pad with the *last observed state* from the plan segment that fit into the context
                last_observed_state_in_context = plan_states_np[num_plan_steps_for_context - 1]
                num_past_padding = self.context_length - num_plan_steps_for_context

                padding_values = np.tile(last_observed_state_in_context, (num_past_padding, 1))
                past_values_np[num_plan_steps_for_context:] = padding_values
                # The mask for these padded values (past_observed_mask_np[num_plan_steps_for_context:]) correctly remains 0.0 as initialized.
        else:
            if self.context_length > 0:
                # Pad with a neutral value, e.g., the initial state if available and meaningful, or zeros.
                padding_values = np.tile(initial_state_np, (self.context_length, 1))
                past_values_np[:] = padding_values

        # Future values and mask
        future_values_np = np.zeros((self.prediction_length, self.state_dim), dtype=np.float32)
        future_observed_mask_np = np.zeros((self.prediction_length, self.state_dim), dtype=np.float32)

        # Actual plan steps that fall into the future window
        # These are states from the original plan starting from index self.context_length
        target_future_plan_states = []
        if plan_len > self.context_length:
            target_future_plan_states = plan_states_np[self.context_length :]

        num_actual_future_steps_from_plan = len(target_future_plan_states)
        len_to_copy_to_future = min(num_actual_future_steps_from_plan, self.prediction_length)

        if len_to_copy_to_future > 0:
            future_values_np[:len_to_copy_to_future] = target_future_plan_states[:len_to_copy_to_future]
            future_observed_mask_np[:len_to_copy_to_future, :] = 1.0

        # Fill remaining future_values with goal_state padding
        if len_to_copy_to_future < self.prediction_length:
            num_future_padding = self.prediction_length - len_to_copy_to_future
            padding_values = np.tile(goal_state_np, (num_future_padding, 1))
            future_values_np[len_to_copy_to_future:] = padding_values
            future_observed_mask_np[len_to_copy_to_future:, :] = 1.0  # Enforce goal prediction

        # Static categorical values (goal state)
        static_categorical_values_np = goal_state_np

        return {
            "freq_token": torch.zeros(1, dtype=torch.long).to(DEVICE),
            "past_values": torch.tensor(past_values_np, dtype=torch.float32).to(DEVICE),
            "future_values": torch.tensor(future_values_np, dtype=torch.float32).to(DEVICE),
            "past_observed_mask": torch.tensor(past_observed_mask_np, dtype=torch.float32).to(DEVICE),
            "future_observed_mask": torch.tensor(future_observed_mask_np, dtype=torch.float32).to(DEVICE),
            "static_categorical_values": torch.tensor(static_categorical_values_np, dtype=torch.float32).to(DEVICE),
        }


# ** Model Class **
class BlocksWorldTTM:
    def __init__(self, model_config: ModelConfig, device: torch.device, output_dir: PosixPath | Path):
        self.config = model_config
        self.device: torch.device = device
        self.model: PreTrainedModel
        self.model_name: str
        self.trainer: Trainer
        self.output_dir: PosixPath | Path = output_dir

    def train(
        self,
        train_dataset,
        val_dataset: Optional[Dataset] = None,
    ):
        logger.info("Starting model training...")

        self.config.state_dim = (
            getattr(getattr(train_dataset, "dataset"), "state_dim")
            if hasattr(train_dataset, "dataset")
            else getattr(train_dataset, "state_dim")
        )

        logger.info(f"Initializing TTM model from: {self.config.ttm_model_path}")

        get_model_params = {
            "model_path": self.config.ttm_model_path,
            "context_length": self.config.context_length,
            "prediction_length": self.config.prediction_length,
            "head_dropout": 0.1,
        }
        self.model: PreTrainedModel = get_model(**get_model_params).to(self.device)

        # Find model name key for logging
        get_model_params["return_model_key"] = True
        self.model_name = str(get_model(**get_model_params))  # Ensure it's string
        logger.info(f"Base TTM model key: {self.model_name}")

        training_args = TrainingArguments(
            output_dir=posixpath.join(self.output_dir, "training_output"),
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            eval_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            seed=self.config.seed,
            report_to="none",
            dataloader_pin_memory=False,
        )

        # Callbacks
        callbacks = [
            TrackingCallback(),
            EarlyStoppingCallback(early_stopping_patience=5),
        ]

        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate)

        # Calculate steps_per_epoch carefully, especially with SubsetRandomSampler
        num_train_samples = len(train_dataset)
        steps_per_epoch = math.ceil(num_train_samples / (self.config.batch_size * training_args.world_size))

        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.config.learning_rate,
            epochs=self.config.num_epochs,
            steps_per_epoch=steps_per_epoch,
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=callbacks,
            optimizers=(optimizer, scheduler),
        )

        logger.info("Trainer initialized. Starting training...")
        self.trainer.train()
        logger.info("Training finished.")

    def predict(self, initial_states: torch.Tensor, goal_states: torch.Tensor) -> torch.Tensor:
        """Generate action sequences to reach goals from given states"""
        if self.model is None:
            raise RuntimeError("Model needs to be trained or loaded before prediction.")
        if self.config.state_dim is None:
            raise RuntimeError("Model config state_dim is not set.")

        self.model.eval()
        with torch.no_grad():
            batch_size = initial_states.shape[0]

            # Ensure inputs are on the correct device
            initial_states = initial_states.to(self.device)
            goal_states = goal_states.to(self.device)

            context_sequence = initial_states.unsqueeze(1).repeat(1, self.config.context_length, 1)

            inputs = {
                "past_values": context_sequence.to(self.device),
                "past_observed_mask": torch.ones_like(context_sequence).to(self.device),
                "static_categorical_values": goal_states,
                "freq_token": torch.zeros(batch_size, dtype=torch.long).to(self.device),
            }

            # Generate predictions
            outputs = self.model(**inputs)
            predictions = torch.sigmoid(outputs[0])
            predictions = torch.round(predictions)

        return predictions

    def save(self, path: Path):
        """Save model weights and configuration"""
        if self.model is None:
            raise RuntimeError("No model to save. Train or load a model first.")

        path.mkdir(parents=True, exist_ok=True)
        model_path = path / "model.pt"
        torch.save(self.model.state_dict(), model_path)

        config_path = path / "config.json"
        with open(config_path, "w") as f:
            json.dump(asdict(self.config), f, indent=4)

        logger.info(f"Model and config saved to {path}")

    @classmethod
    def load(cls, path: Path, device: torch.device) -> "BlocksWorldTTM":
        """Load model weights and configuration"""
        logger.info(f"Loading model from {path}")
        config_path = path / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config_dict = json.load(f)

        # Ensure all required fields for ModelConfig are present
        try:
            loaded_model_config = ModelConfig(**config_dict)
        except TypeError as e:
            logger.error(f"Error loading ModelConfig from {config_path}. Missing fields or mismatch: {e}")
            logger.error(f"Loaded config_dict: {config_dict}")
            raise

        if loaded_model_config.state_dim is None:
            raise ValueError("Loaded model config must contain 'state_dim'.")

        instance = cls(model_config=loaded_model_config, device=device, output_dir=path)

        get_model_params = {
            "model_path": loaded_model_config.ttm_model_path,
            "context_length": loaded_model_config.context_length,
            "prediction_length": loaded_model_config.prediction_length,
            "head_dropout": 0.1,
        }
        instance.model = get_model(**get_model_params).to(instance.device)

        model_path = path / "model.pt"
        if not model_path.is_file():
            raise FileNotFoundError(f"Model weights file not found: {model_path}")

        instance.model.load_state_dict(torch.load(model_path, map_location=instance.device))
        instance.model.eval()

        # Try to get model_name
        try:
            get_model_params["return_model_key"] = True
            instance.model_name = str(get_model(**get_model_params))
        except Exception:
            instance.model_name = Path(instance.config.ttm_model_path).name

        logger.info(f"Model loaded successfully from {path}. Base TTM: {instance.model_name}")
        return instance


# ** Helper Functions **
def setup_logging(level="INFO", log_file: Optional[Path] = None):
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        level=level.upper(),
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            rotation="10 MB",
            level="DEBUG",  # Log DEBUG to file
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        )
    logger.info(f"Logging setup complete. Level: {level}. File: {log_file}")


def analyze_dataset(data_path: Path) -> Dict[str, Any]:
    logger.info(f"Analyzing dataset: {data_path}")
    with open(data_path, "r") as f:
        data = json.load(f)["plans"]

    if not data:
        logger.error(f"No plans found in dataset {data_path}.")
        raise ValueError("Empty dataset.")

    max_plan_length = max(len(item["plan"]) for item in data)
    avg_plan_length = sum(len(item["plan"]) for item in data) / len(data)
    state_dim = len(data[0]["initial_state"])
    num_samples = len(data)

    stats = {
        "max_plan_length": max_plan_length,
        "avg_plan_length": avg_plan_length,
        "state_dim": state_dim,
        "num_samples": num_samples,
        "recommended_prediction_length": max_plan_length + 2,  # Small buffer
    }

    logger.info(f"Dataset Statistics:\n{pformat(stats)}")
    return stats


def determine_ttm_lengths(
    max_plan_length: int,
    recommended_prediction_length: int,
    user_context_length: Optional[int] = None,
    user_prediction_length: Optional[int] = None,
) -> Tuple[int, int]:
    # First, find the context length
    final_context_length = user_context_length
    valid_cls = [cl for cl in SUPPORTED_CONTEXT_LENGTHS if cl <= max_plan_length]
    if valid_cls:
        final_context_length = max(valid_cls)
    else:
        final_context_length = min(SUPPORTED_CONTEXT_LENGTHS)
        logger.warning(
            f"Max plan length ({max_plan_length}) is smaller than all supported context lengths. "
            f"Choosing the smallest supported: {final_context_length}."
        )
    logger.info(f"Max plan length: {max_plan_length} | Auto-selected context_length: {final_context_length}")

    # Secondly, find the prediction length
    final_prediction_length = user_prediction_length
    valid_fls = [fl for fl in SUPPORTED_PREDICTION_LENGTHS if fl <= recommended_prediction_length]
    if valid_fls:
        final_prediction_length = max(valid_fls)
    else:
        final_prediction_length = min(SUPPORTED_PREDICTION_LENGTHS)
        logger.warning(
            f"Recommended prediction length ({recommended_prediction_length}) is smaller than all supported forecast lengths. "
            f"Choosing the smallest supported: {final_prediction_length}."
        )
    logger.info(
        f"Recommended prediction length: {recommended_prediction_length} | Auto-selected prediction_length: {final_prediction_length}"
    )

    return final_context_length, final_prediction_length


def prepare_datasets(
    data_path: Path, context_length: int, prediction_length: int, seed: int
) -> Tuple[Dataset, Dataset, Dataset]:
    logger.info("Preparing train/validation/test datasets...")
    full_dataset = BlocksWorldDataset(str(data_path), context_length, prediction_length)

    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    logger.info(f"Splitting dataset: Total={total_size}, Train={train_size}, Val={val_size}, Test={test_size}")

    if train_size + val_size + test_size != total_size:
        raise ValueError(
            f"Dataset split sizes do not match total size. "
            f"Train: {train_size}, Val: {val_size}, Test: {test_size}, Total: {total_size}"
        )

    else:
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(seed),
        )

    logger.info(
        f"Dataset split complete. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )
    return train_dataset, val_dataset, test_dataset


def evaluate_model(model: BlocksWorldTTM, test_dataset) -> Dict[str, Any]:
    """Comprehensive evaluation of the model with more detailed metrics"""
    logger.info("Starting model evaluation on test set...")
    if model.model is None:
        logger.error("Model not loaded or trained in BlocksWorldTTM instance.")
        return {}
    model.model.eval()
    all_predictions = []
    all_targets = []
    goal_state_predictions = []
    goal_state_targets = []

    num_samples = len(test_dataset)
    if num_samples == 0:
        logger.warning("Test dataset is empty. Skipping evaluation.")
        return {"num_samples": 0}

    num_exact_matches = 0
    num_partial_matches = 0
    total_bits_correct = 0
    total_bits = 0

    with torch.no_grad():
        for i in range(num_samples):
            sample = test_dataset[i]

            # Get initial and goal states
            initial_state = sample["past_values"][0]
            goal_state = sample["static_categorical_values"]
            target = sample["future_values"]

            # Create context sequence
            context_sequence = initial_state.unsqueeze(0).repeat(1, model.config.context_length, 1)

            # Prepare inputs
            inputs = {
                "past_values": context_sequence.to(model.device),
                "past_observed_mask": torch.ones_like(context_sequence).to(model.device),
                "static_categorical_values": goal_state.unsqueeze(0).to(model.device),
                "freq_token": torch.zeros(1, dtype=torch.long).to(model.device),
            }

            # Get prediction
            outputs = model.model(**inputs)
            prediction = torch.sigmoid(outputs[0])
            prediction = torch.round(prediction)

            # Store predictions and targets
            all_predictions.append(prediction)
            all_targets.append(target)

            # Focus on goal states (final states)
            pred_goal = prediction[0, -1]
            true_goal = target[-1]

            goal_state_predictions.append(pred_goal)
            goal_state_targets.append(true_goal)

            # Calculate exact matches
            if torch.all(pred_goal == true_goal):
                num_exact_matches += 1

            # Calculate partial matches (more than 50% bits correct)
            num_correct_bits = torch.sum(pred_goal == true_goal).item()
            total_bits_correct += num_correct_bits
            total_bits += len(pred_goal)

            if num_correct_bits > len(pred_goal) / 2:
                num_partial_matches += 1

    # Calculate metrics
    metrics = {
        "num_samples": num_samples,
        "num_exact_matches": num_exact_matches,
        "exact_match_rate": num_exact_matches / num_samples,
        "num_partial_matches": num_partial_matches,  # >50% correct
        "partial_match_rate": num_partial_matches / num_samples,
        "bit_accuracy": total_bits_correct / total_bits,
    }

    logger.info("Detailed Model Evaluation Metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    return metrics


def analyze_error_patterns(model: BlocksWorldTTM, test_dataset) -> Dict[str, Any]:
    logger.info("Analyzing error patterns...")
    if model.model is None:
        logger.error("Model not loaded or trained in BlocksWorldTTM instance.")
        return {}
    model.model.eval()

    successes = []
    failures = []
    bit_error_counts: Dict[int, int] = {}  # Tracks which bit indices are most commonly wrong

    num_samples = len(test_dataset)
    if num_samples == 0:
        logger.warning("Test dataset is empty. Skipping error analysis.")
        return {
            "num_successes": 0,
            "num_failures": 0,
            "success_rate": 0,
            "bit_error_counts": {},
            "success": [],
            "failure": [],
        }

    with torch.no_grad():
        for i in range(len(test_dataset)):
            sample = test_dataset[i]

            # Get initial and goal states
            initial_state_tensor = sample["past_values"][0]
            goal_state_tensor = sample["static_categorical_values"]
            target_tensor = sample["future_values"][-1]

            # Create context sequence
            context_sequence = initial_state_tensor.unsqueeze(0).repeat(1, model.config.context_length, 1)

            # Prepare inputs
            inputs = {
                "past_values": context_sequence.to(model.device),
                "past_observed_mask": torch.ones_like(context_sequence).to(model.device),
                "static_categorical_values": goal_state_tensor.unsqueeze(0).to(model.device),
                "freq_token": torch.zeros(1, dtype=torch.long).to(model.device),
            }

            # Get prediction
            outputs = model.model(**inputs)
            prediction = torch.sigmoid(outputs[0])
            prediction = torch.round(prediction)
            predicted_goal_tensor = prediction[0, -1]

            # Calculate error statistics
            errors_tensor = (predicted_goal_tensor != target_tensor).nonzero().squeeze(1)
            num_errors = len(errors_tensor)

            # Track which bits had errors
            for error_idx in errors_tensor:
                bit_index = error_idx.item()
                if bit_index not in bit_error_counts:
                    bit_error_counts[bit_index] = 0
                bit_error_counts[bit_index] += 1

            # Convert tensors to NumPy arrays and then to Python lists for JSON serialization
            case = {
                "initial_state": initial_state_tensor.cpu().numpy().tolist(),
                "goal_state": goal_state_tensor.cpu().numpy().tolist(),
                "predicted_goal": predicted_goal_tensor.cpu().numpy().tolist(),
                "target_goal": target_tensor.cpu().numpy().tolist(),
                "num_errors": num_errors,  # This is an int, fine
                "error_positions": errors_tensor.cpu().numpy().tolist(),
            }

            if num_errors == 0:
                successes.append(case)
            else:
                failures.append(case)

    total_cases = len(successes) + len(failures)
    success_rate = (len(successes) / total_cases) if total_cases > 0 else 0.0

    analysis = {
        "num_successes": len(successes),
        "num_failures": len(failures),
        "success_rate": success_rate,
        "bit_error_counts": bit_error_counts,
        "success": successes,
        "failure": failures,
    }

    logger.info("Error Pattern Analysis:")
    logger.info(f"  Number of successful predictions (exact final state match): {analysis['num_successes']}")
    logger.info(f"  Number of failed predictions: {analysis['num_failures']}")
    logger.info(f"  Success rate: {analysis['success_rate']:.4f}")

    if bit_error_counts:
        logger.info("  Most common error positions (bit_index: count):")
        sorted_errors = sorted(bit_error_counts.items(), key=lambda x: x[1], reverse=True)
        for bit, count in sorted_errors[:5]:  # Log top 5
            logger.info(f"    Bit {bit}: {count} errors")

    # Log example successes/failures
    logger.info("\nExample Successes (up to 3):")
    for i, case_data in enumerate(analysis["success"][:3]):
        logger.info(f"Success Case {i + 1}:")
        logger.info(f"  Initial State: {case_data['initial_state']}")
        logger.info(f"  Goal State: {case_data['goal_state']}")
        logger.info(f"  Predicted Final State: {case_data['predicted_goal']}")

    logger.info("\nExample Failures (up to 3):")
    for i, case_data in enumerate(analysis["failure"][:3]):
        logger.info(f"Failure Case {i + 1}:")
        logger.info(f"  Initial State: {case_data['initial_state']}")
        logger.info(f"  Goal State: {case_data['goal_state']}")
        logger.info(f"  Predicted Final State: {case_data['predicted_goal']}")
        logger.info(f"  Target Final State: {case_data['target_goal']}")
        logger.info(f"  Number of Errors: {case_data['num_errors']}")
        logger.info(f"  Error Positions: {case_data['error_positions']}")

    return analysis


# ** Main Execution **
def main():
    parser = argparse.ArgumentParser(description="Train or evaluate a TTM model on the BlocksWorld domain.")
    parser.add_argument("mode", choices=["train", "evaluate"], help="Script mode: train or evaluate.")
    parser.add_argument("--dataset_file", type=Path, required=True, help="Path to the JSON dataset file.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output_blocksworld_ttm"),
        help="Directory to save models, logs, and results.",
    )
    parser.add_argument(
        "--model_load_path",
        type=Path,
        help="Path to load a pre-trained model's assets directory (required for 'evaluate' mode).",
    )

    # Model and training parameters
    parser.add_argument(
        "--ttm_model_path",
        type=str,
        default=DEFAULT_TTM_MODEL_PATH,
        help="Base TTM model path from HuggingFace or local. Default to 'ibm-granite/granite-timeseries-ttm-r2'.",
    )
    parser.add_argument(
        "--context_length",
        type=int,
        help="Context length for TTM. If not provided, determined from dataset.",
    )
    parser.add_argument(
        "--prediction_length",
        type=int,
        help="Prediction length for TTM. If not provided, determined from dataset.",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate. Default is 1e-4.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size. Default is 32.")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs. Default is 50.")

    # Other parameters
    parser.add_argument("--seed", type=int, default=13, help="Random seed. Default is 13.")
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level. Default is INFO.",
    )

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_file = args.output_dir / f"run_{args.mode}.log"
    setup_logging(args.log_level, log_file)

    logger.info(f"Script arguments: {pformat(vars(args))}")

    # Set seed
    set_seed(args.seed)
    logger.info(f"Random seed set to: {args.seed}")

    # Infer the number of blocks
    num_blocks = int(str(args.dataset_file).split("_")[-1][0])
    logger.info(f"Inferred number of blocks from dataset filename: {num_blocks}")

    if args.mode == "train":
        logger.info("** Starting Training Mode **")

        dataset_stats = analyze_dataset(args.dataset_file)

        auto_context_length, auto_prediction_length = determine_ttm_lengths(
            dataset_stats["max_plan_length"],
            dataset_stats["recommended_prediction_length"],
            args.context_length,
            args.prediction_length,
        )

        model_cfg = ModelConfig(
            context_length=auto_context_length,
            prediction_length=auto_prediction_length,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            ttm_model_path=args.ttm_model_path,
            seed=args.seed,
            state_dim=dataset_stats["state_dim"],
        )
        logger.info(f"Using ModelConfig: {pformat(asdict(model_cfg))}")

        train_ds, val_ds, test_ds = prepare_datasets(
            args.dataset_file, model_cfg.context_length, model_cfg.prediction_length, args.seed
        )

        if not train_ds:
            logger.error("Training dataset is empty or None. Cannot train.")
            return 1

        ttm_model = BlocksWorldTTM(model_cfg, device=DEVICE, output_dir=args.output_dir)
        ttm_model.train(train_ds, val_ds)

        final_model_assets_path = args.output_dir / "final_model_assets"
        if final_model_assets_path.exists():
            logger.error(
                f"Model save path {final_model_assets_path} already exists."
                f"\nSuffixing `final_model_assets` with current timestamp (formatted nicely) to avoid overwriting."
            )
            final_model_assets_path = final_model_assets_path.with_name(
                f"final_model_assets_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

        logger.info(f"Saving model assets to {final_model_assets_path}")
        ttm_model.save(final_model_assets_path)

        if test_ds:
            logger.info("** Evaluating model on test set after training **")
            metrics = evaluate_model(ttm_model, test_ds)
            with open(args.output_dir / "test_metrics_after_train.json", "w") as f:
                json.dump(metrics, f, indent=4)

            error_analysis = analyze_error_patterns(ttm_model, test_ds)
            with open(args.output_dir / "test_error_analysis_after_train.json", "w") as f:
                json.dump(error_analysis, f, indent=4)

            logger.info(f"Post-training evaluation results saved to {args.output_dir}")
        else:
            logger.info("No test dataset available for evaluation after training.")

    elif args.mode == "evaluate":
        logger.info("** Starting Evaluation Mode **")
        if not args.model_load_path:
            logger.error("--model_load_path is required for 'evaluate' mode.")
            return 1
        if not args.model_load_path.exists() or not args.model_load_path.is_dir():
            logger.error(f"Model assets path {args.model_load_path} does not exist or is not a directory.")
            return 1

        ttm_model = BlocksWorldTTM.load(args.model_load_path, device=DEVICE)

        # state_dim from loaded model's config is used by BlocksWorldTTM.load
        # context_length and prediction_length also from loaded config
        _, _, test_ds = prepare_datasets(  # We only need test_ds here
            args.dataset_file,
            ttm_model.config.context_length,
            ttm_model.config.prediction_length,
            args.seed,
        )

        if not test_ds:
            logger.error("Test dataset is empty or None. Cannot evaluate.")
            return 1

        eval_results_dir = args.output_dir / f"evaluation_results_{args.model_load_path.name}"
        eval_results_dir.mkdir(parents=True, exist_ok=True)

        metrics = evaluate_model(ttm_model, test_ds)
        with open(eval_results_dir / "evaluation_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        error_analysis = analyze_error_patterns(ttm_model, test_ds)
        with open(eval_results_dir / "evaluation_error_analysis.json", "w") as f:
            json.dump(error_analysis, f, indent=4)

        logger.info(f"Evaluation results saved to {eval_results_dir}")

    logger.info("Script finished.")
    return 0


if __name__ == "__main__":
    main()
