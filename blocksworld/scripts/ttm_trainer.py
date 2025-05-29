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
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torch.utils.tensorboard.writer import SummaryWriter
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer import Trainer
from transformers.trainer_callback import EarlyStoppingCallback, TrainerCallback
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


# Callback class for custom TensorBoard logging
class TensorBoardLoggingCallback(TrainerCallback):
    def __init__(self, tb_writer: SummaryWriter):
        self.tb_writer = tb_writer

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Called when logging occurs during training"""
        if logs is not None and self.tb_writer is not None:
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    # Log all numeric metrics
                    self.tb_writer.add_scalar(f"training/{key}", value, state.global_step)

    def on_evaluate(self, args, state, control, model=None, logs=None, **kwargs):
        """Called after evaluation"""
        if logs is not None and self.tb_writer is not None:
            for key, value in logs.items():
                if isinstance(value, (int, float)) and key.startswith("eval_"):
                    # Log evaluation metrics
                    self.tb_writer.add_scalar(f"evaluation/{key}", value, state.global_step)


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

    def _scale_binary_array(self, data_array_np: np.ndarray) -> np.ndarray:
        """Scales a numpy array of 0s and 1s to -1s and 1s."""
        return data_array_np * 2.0 - 1.0

    def __len__(self):  # Length of the Dataset
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        plan_states_np_orig = np.array(sample.plan, dtype=np.float32)  # 0/1
        goal_state_np_orig = np.array(sample.goal_state, dtype=np.float32)  # 0/1

        # initial_state_np_orig will be plan_states_np_orig[0] if plan is not empty
        # Handle empty plan case for initial_state_np_orig if necessary, though current logic assumes plan[0] is initial.
        # For safety, let's ensure initial_state_np_orig is explicitly defined.
        if len(sample.plan) > 0:
            initial_state_np_orig = np.array(sample.plan[0], dtype=np.float32)
        else:  # Should not happen with current plan generator, but good for robustness
            initial_state_np_orig = np.array(sample.initial_state, dtype=np.float32)

        # Past values and mask (populated with 0/1)
        past_values_np = np.zeros((self.context_length, self.state_dim), dtype=np.float32)
        past_observed_mask_np = np.zeros((self.context_length, self.state_dim), dtype=np.float32)

        # Number of actual plan steps to copy into the context window
        num_plan_steps_for_context = 0
        if len(plan_states_np_orig) > 0:
            num_plan_steps_for_context = min(len(plan_states_np_orig), self.context_length)

        if num_plan_steps_for_context > 0:
            # Copy the actual plan steps
            past_values_np[:num_plan_steps_for_context] = plan_states_np_orig[:num_plan_steps_for_context]
            past_observed_mask_np[:num_plan_steps_for_context, :] = 1.0

            # If padding is needed for the rest of the context window
            if num_plan_steps_for_context < self.context_length:
                # Pad with the *last observed state* from the plan segment that fit into the context
                last_observed_state_in_context = plan_states_np_orig[num_plan_steps_for_context - 1]
                num_past_padding = self.context_length - num_plan_steps_for_context

                padding_values = np.tile(last_observed_state_in_context, (num_past_padding, 1))
                past_values_np[num_plan_steps_for_context:] = padding_values
                # The mask for these padded values (past_observed_mask_np[num_plan_steps_for_context:]) correctly remains 0.0 as initialized.
        elif self.context_length > 0:  # No plan steps, context length > 0
            # Pad with a neutral value, e.g., the initial state if available and meaningful, or zeros.
            padding_values = np.tile(initial_state_np_orig, (self.context_length, 1))
            past_values_np[:] = padding_values
            # past_observed_mask_np remains 0.0 for these padded initial states

        # Future values and mask (populated with 0/1)
        future_values_np = np.zeros((self.prediction_length, self.state_dim), dtype=np.float32)
        future_observed_mask_np = np.zeros((self.prediction_length, self.state_dim), dtype=np.float32)

        target_future_plan_states_orig = []
        if len(plan_states_np_orig) > self.context_length:  # If plan extends beyond context
            target_future_plan_states_orig = plan_states_np_orig[self.context_length :]

        num_actual_future_steps_from_plan = len(target_future_plan_states_orig)
        len_to_copy_to_future = min(num_actual_future_steps_from_plan, self.prediction_length)

        if len_to_copy_to_future > 0:
            future_values_np[:len_to_copy_to_future] = target_future_plan_states_orig[:len_to_copy_to_future]
            future_observed_mask_np[:len_to_copy_to_future, :] = 1.0

        if len_to_copy_to_future < self.prediction_length:
            num_future_padding = self.prediction_length - len_to_copy_to_future
            padding_values = np.tile(goal_state_np_orig, (num_future_padding, 1))
            future_values_np[len_to_copy_to_future:] = padding_values
            future_observed_mask_np[len_to_copy_to_future:, :] = 1.0

        static_categorical_values_np = goal_state_np_orig.copy()  # 0/1

        # ** Now, scale the constructed numpy arrays to -1/1 **
        past_values_np_scaled = self._scale_binary_array(past_values_np)
        future_values_np_scaled = self._scale_binary_array(future_values_np)
        static_categorical_values_np_scaled = self._scale_binary_array(static_categorical_values_np)

        return {
            "freq_token": torch.zeros(1, dtype=torch.long).to(DEVICE),
            "past_values": torch.tensor(past_values_np_scaled, dtype=torch.float32).to(DEVICE),
            "future_values": torch.tensor(future_values_np_scaled, dtype=torch.float32).to(DEVICE),
            "past_observed_mask": torch.tensor(past_observed_mask_np, dtype=torch.float32).to(DEVICE),
            "future_observed_mask": torch.tensor(future_observed_mask_np, dtype=torch.float32).to(DEVICE),
            "static_categorical_values": torch.tensor(static_categorical_values_np_scaled, dtype=torch.float32).to(
                DEVICE
            ),
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
        self.tb_writer: Optional[SummaryWriter] = None

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

        # Initialize TensorBoard writer
        tensorboard_log_dir = self.output_dir / "tensorboard_logs"
        tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=str(tensorboard_log_dir))
        logger.info(f"TensorBoard logs will be saved to: {tensorboard_log_dir}")

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
            report_to="tensorboard",  # Enable TensorBoard reporting
            logging_dir=str(tensorboard_log_dir),  # Set logging directory
            logging_strategy="steps",
            logging_steps=10,  # Log every 10 steps
            dataloader_pin_memory=False,
        )

        # Callbacks
        callbacks = [
            TrackingCallback(),
            EarlyStoppingCallback(early_stopping_patience=5),
            TensorBoardLoggingCallback(self.tb_writer),  # Custom callback for additional logging
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

        # Log hyperparameters to TensorBoard
        self.tb_writer.add_hparams(
            {
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "num_epochs": self.config.num_epochs,
                "context_length": self.config.context_length,
                "prediction_length": self.config.prediction_length,
                "state_dim": self.config.state_dim,
            },
            {},
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

        # Close TensorBoard writer
        if self.tb_writer:
            self.tb_writer.close()

    def predict(self, initial_states: torch.Tensor, goal_states: torch.Tensor) -> torch.Tensor:
        """Generate action sequences to reach goals from given states.
        initial_states and goal_states are expected to be 0/1.
        The method will scale them internally before feeding to the model.
        Returns predictions in 0/1 format.
        """
        if self.model is None:
            raise RuntimeError("Model needs to be trained or loaded before prediction.")
        if self.config.state_dim is None:
            raise RuntimeError("Model config state_dim is not set.")

        self.model.eval()
        with torch.no_grad():
            batch_size = initial_states.shape[0]

            # Scale inputs from 0/1 to -1/1
            initial_states_scaled = initial_states.to(self.device) * 2.0 - 1.0
            goal_states_scaled = goal_states.to(self.device) * 2.0 - 1.0

            # Context sequence: repeat the scaled initial state
            context_sequence_scaled = initial_states_scaled.unsqueeze(1).repeat(1, self.config.context_length, 1)

            inputs = {
                "past_values": context_sequence_scaled,
                "past_observed_mask": torch.ones_like(context_sequence_scaled).to(
                    self.device
                ),  # Mask is 1 for observed
                "static_categorical_values": goal_states_scaled,
                "freq_token": torch.zeros(batch_size, dtype=torch.long).to(self.device),
            }

            outputs = self.model(**inputs)
            raw_logits = outputs[0]  # Shape: (batch_size, prediction_length, num_features)

            predictions_tanh = torch.tanh(raw_logits)

            # Binarize to -1 or 1 based on tanh output (threshold at 0)
            predictions_scaled_binary = torch.where(
                predictions_tanh > 0, torch.tensor(1.0, device=self.device), torch.tensor(-1.0, device=self.device)
            )

            # Convert back to 0/1 for the return value
            predictions_original_binary = (predictions_scaled_binary + 1.0) / 2.0

        return predictions_original_binary

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
            goal_state = sample["static_categorical_values"]  # This is already -1/1
            target = sample["future_values"]  # This is already -1/1

            # Create context sequence
            context_sequence = initial_state.unsqueeze(0).repeat(
                1, model.config.context_length, 1
            )  # initial_state is -1/1

            # Prepare inputs
            inputs = {
                "past_values": context_sequence.to(model.device),  # context_sequence is -1/1
                "past_observed_mask": torch.ones_like(context_sequence).to(model.device),
                "static_categorical_values": goal_state.unsqueeze(0).to(model.device),  # goal_state is -1/1
                "freq_token": torch.zeros(1, dtype=torch.long).to(model.device),
            }

            # Get prediction
            outputs = model.model(**inputs)
            raw_logits = outputs[0]
            predictions_tanh = torch.tanh(raw_logits)
            prediction_scaled_binary = torch.where(
                predictions_tanh > 0, torch.tensor(1.0, device=model.device), torch.tensor(-1.0, device=model.device)
            )

            # target is already scaled from the dataset
            target_scaled = target.to(model.device)

            # Focus on goal states (final states) in -1/1 space
            pred_goal_scaled = prediction_scaled_binary[0, -1]  # Last step of prediction
            true_goal_scaled = target_scaled[-1]  # Last step of target (should be goal if padded)

            # goal_state_predictions.append(pred_goal)
            # goal_state_targets.append(true_goal)

            # Calculate exact matches
            if torch.all(pred_goal_scaled == true_goal_scaled):
                num_exact_matches += 1

            # Calculate partial matches (more than 50% bits correct)
            num_correct_bits = torch.sum(pred_goal_scaled == true_goal_scaled).item()
            total_bits_correct += num_correct_bits
            total_bits += len(pred_goal_scaled)

            if num_correct_bits > len(pred_goal_scaled) / 2:
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


def analyze_error_patterns(model: BlocksWorldTTM, test_dataset: Dataset) -> Dict[str, Any]:
    logger.info("Analyzing error patterns (data is -1/1)...")
    if model.model is None:
        logger.error("Model not loaded or trained in BlocksWorldTTM instance.")
        return {}
    model.model.eval()

    successes = []
    failures = []
    # Tracks which bit indices (0-based) are most commonly wrong
    # The keys will be the feature indices.
    bit_error_counts: Dict[int, int] = {}

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
            sample = test_dataset[i]  # Data from BlocksWorldDataset is already -1/1

            # Get initial, goal, and target states (all should be -1/1)
            initial_state_tensor = sample["past_values"][0].to(model.device)  # (state_dim,)
            goal_state_tensor = sample["static_categorical_values"].to(model.device)  # (state_dim,)
            # Target is the full future sequence; we're interested in the final step for goal comparison
            target_final_state_tensor = sample["future_values"][-1].to(model.device)  # (state_dim,)

            # Create context sequence (repeating the scaled initial state)
            # initial_state_tensor is already scaled (-1/1)
            context_sequence = initial_state_tensor.unsqueeze(0).repeat(1, model.config.context_length, 1)

            # Prepare inputs for the model
            inputs = {
                "past_values": context_sequence,  # Already -1/1
                "past_observed_mask": torch.ones_like(context_sequence).to(model.device),
                "static_categorical_values": goal_state_tensor.unsqueeze(0),  # Already -1/1, add batch dim
                "freq_token": torch.zeros(1, dtype=torch.long).to(model.device),
            }

            # Get prediction
            outputs = model.model(**inputs)
            raw_logits = outputs[0]  # Shape: (1, prediction_length, state_dim)

            predictions_tanh = torch.tanh(raw_logits)
            # Binarize to -1 or 1 based on tanh output (threshold at 0)
            predictions_scaled_binary = torch.where(
                predictions_tanh > 0, torch.tensor(1.0, device=model.device), torch.tensor(-1.0, device=model.device)
            )

            # Get the predicted final state from the sequence
            predicted_final_state_tensor = predictions_scaled_binary[0, -1, :]  # (state_dim,)

            # Calculate error statistics by comparing predicted_final_state_tensor with target_final_state_tensor
            # Both are in -1/1 format
            errors_mask = predicted_final_state_tensor != target_final_state_tensor
            errors_indices_tensor = errors_mask.nonzero(as_tuple=False).squeeze(1)  # Get indices of True values

            num_errors = errors_indices_tensor.numel()

            # Track which bits had errors
            for error_idx_tensor in errors_indices_tensor:
                bit_index = error_idx_tensor.item()  # Convert tensor to Python int
                bit_error_counts[bit_index] = bit_error_counts.get(bit_index, 0) + 1

            # Convert tensors to NumPy arrays and then to Python lists for JSON serialization
            # All these tensors are already in -1/1 format.
            case = {
                "initial_state": initial_state_tensor.cpu().numpy().tolist(),
                "goal_state": goal_state_tensor.cpu().numpy().tolist(),
                "predicted_final_state": predicted_final_state_tensor.cpu().numpy().tolist(),
                "target_final_state": target_final_state_tensor.cpu().numpy().tolist(),
                "num_errors": num_errors,
                "error_positions": errors_indices_tensor.cpu()
                .numpy()
                .tolist(),  # List of feature indices that were wrong
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
        "bit_error_counts": bit_error_counts,  # {feature_index: count_of_errors_for_this_feature}
        "success_cases": successes,  # Renamed for clarity
        "failure_cases": failures,  # Renamed for clarity
    }

    logger.info("Error Pattern Analysis (States are -1/1):")
    logger.info(f"  Number of successful predictions (exact final state match): {analysis['num_successes']}")
    logger.info(f"  Number of failed predictions: {analysis['num_failures']}")
    logger.info(f"  Success rate (exact final state match): {analysis['success_rate']:.4f}")

    if bit_error_counts:
        logger.info("  Most common error positions (feature_index: count):")
        # Sort by count descending, then by feature_index ascending for tie-breaking
        sorted_errors = sorted(bit_error_counts.items(), key=lambda x: (-x[1], x[0]))
        for feature_idx, count in sorted_errors[:10]:  # Log top 10
            logger.info(f"    Feature Index {feature_idx}: {count} errors")
    else:
        logger.info("  No bit errors recorded across all failed predictions (or no failures).")

    # Helper to convert -1/1 state to 0/1 string for logging
    def format_state_for_log(state_list_minus_one_one):
        if not state_list_minus_one_one:
            return "[]"
        # Convert to 0/1 for easier reading in logs
        state_01 = [int((x + 1) / 2) for x in state_list_minus_one_one]
        return str(state_01)

    logger.info("\nExample Successes (up to 3, states shown as 0/1 for readability):")
    for i, case_data in enumerate(analysis["success_cases"][:3]):
        logger.info(f"Success Case {i + 1}:")
        logger.info(f"  Initial State (0/1): {format_state_for_log(case_data['initial_state'])}")
        logger.info(f"  Goal State (0/1):    {format_state_for_log(case_data['goal_state'])}")
        logger.info(f"  Predicted Final (0/1): {format_state_for_log(case_data['predicted_final_state'])}")
        # Target final state is same as predicted for success cases
        # logger.info(f"  Target Final (0/1):    {format_state_for_log(case_data['target_final_state'])}")

    logger.info("\nExample Failures (up to 3, states shown as 0/1 for readability):")
    for i, case_data in enumerate(analysis["failure_cases"][:3]):
        logger.info(f"Failure Case {i + 1}:")
        logger.info(f"  Initial State (0/1): {format_state_for_log(case_data['initial_state'])}")
        logger.info(f"  Goal State (0/1):    {format_state_for_log(case_data['goal_state'])}")
        logger.info(f"  Predicted Final (0/1): {format_state_for_log(case_data['predicted_final_state'])}")
        logger.info(f"  Target Final (0/1):    {format_state_for_log(case_data['target_final_state'])}")
        logger.info(f"  Number of Errors: {case_data['num_errors']}")
        logger.info(f"  Error Positions (feature indices): {case_data['error_positions']}")

    return analysis


def prepare_overfit_test_datasets(
    data_path: Path, context_length: int, prediction_length: int, seed: int, num_overfit_samples: int = 4
) -> Tuple[Dataset, Optional[Dataset]]:  # Val dataset can be None for this test
    logger.info(f"Preparing dataset for overfitting test with {num_overfit_samples} samples...")
    full_dataset = BlocksWorldDataset(str(data_path), context_length, prediction_length)

    if len(full_dataset) < num_overfit_samples:
        logger.warning(
            f"Full dataset ({len(full_dataset)}) is smaller than requested num_overfit_samples ({num_overfit_samples}). "
            f"Using all available samples."
        )
        num_overfit_samples = len(full_dataset)
        if num_overfit_samples == 0:
            logger.error("Dataset is empty. Cannot perform overfitting test.")
            return None, None

    # Create a subset of the full_dataset for overfitting
    indices = list(range(num_overfit_samples))
    overfit_train_dataset = Subset(full_dataset, indices)

    logger.info(
        f"Overfit training dataset created with {len(overfit_train_dataset)} samples."
        f" State_dim: {getattr(full_dataset, 'state_dim')}"  # Access state_dim from the original full_dataset
    )

    # For overfitting, we often don't need a separate validation set, as we're just checking if it can memorize the training data.
    overfit_val_dataset = None
    # If the Trainer setup strictly requires it, use the same small set.
    # overfit_val_dataset = Subset(full_dataset, indices)

    return overfit_train_dataset, overfit_val_dataset


# ** Main Execution **
def main_overfit():
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
        logger.info("** Starting Training Mode (Overfitting Test Variant) **")

        dataset_stats = analyze_dataset(args.dataset_file)

        auto_context_length, auto_prediction_length = determine_ttm_lengths(
            dataset_stats["max_plan_length"],
            dataset_stats["recommended_prediction_length"],
            args.context_length,
            args.prediction_length,
        )

        # ** Overfitting Test Specific Config **
        NUM_OVERFIT_SAMPLES = 4  # Or make this an argparse argument
        OVERFIT_EPOCHS = 200  # Train for longer to ensure memorization
        OVERFIT_BATCH_SIZE = min(args.batch_size, NUM_OVERFIT_SAMPLES)  # Batch size can be small
        OVERFIT_LR = args.learning_rate  # Start with the same LR, might need adjustment

        logger.info("** OVERFITTING TEST PARAMETERS **")
        logger.info(f"Number of samples to overfit: {NUM_OVERFIT_SAMPLES}")
        logger.info(f"Epochs for overfitting: {OVERFIT_EPOCHS}")
        logger.info(f"Batch size for overfitting: {OVERFIT_BATCH_SIZE}")
        logger.info(f"Learning rate for overfitting: {OVERFIT_LR}")
        logger.info("**********************")

        model_cfg = ModelConfig(
            context_length=auto_context_length,
            prediction_length=auto_prediction_length,
            learning_rate=OVERFIT_LR,  # Use overfit LR
            batch_size=OVERFIT_BATCH_SIZE,  # Use overfit batch size
            num_epochs=OVERFIT_EPOCHS,  # Use overfit epochs
            ttm_model_path=args.ttm_model_path,
            seed=args.seed,
            state_dim=dataset_stats["state_dim"],
        )
        logger.info(f"Using ModelConfig for Overfitting: {pformat(asdict(model_cfg))}")

        # Use the new dataset preparation function
        train_ds, val_ds = prepare_overfit_test_datasets(
            args.dataset_file,
            model_cfg.context_length,
            model_cfg.prediction_length,
            args.seed,
            num_overfit_samples=NUM_OVERFIT_SAMPLES,
        )

        if not train_ds:
            logger.error("Training dataset for overfitting is empty or None. Cannot train.")
            return 1

        ttm_model = BlocksWorldTTM(model_cfg, device=DEVICE, output_dir=args.output_dir)

        logger.info("Initializing TTM model for overfitting test...")
        get_model_params = {
            "model_path": model_cfg.ttm_model_path,
            "context_length": model_cfg.context_length,
            "prediction_length": model_cfg.prediction_length,
            "head_dropout": 0.1,  # Keep consistent with your main training
        }
        overfit_model_instance: PreTrainedModel = get_model(**get_model_params).to(DEVICE)

        overfit_training_args = TrainingArguments(
            output_dir=posixpath.join(args.output_dir, "overfitting_training_output"),
            learning_rate=model_cfg.learning_rate,
            num_train_epochs=model_cfg.num_epochs,
            per_device_train_batch_size=model_cfg.batch_size,
            per_device_eval_batch_size=model_cfg.batch_size if val_ds else model_cfg.batch_size,
            eval_strategy="no",  # Not strictly needed for overfitting test
            save_strategy="no",
            load_best_model_at_end=False,
            seed=model_cfg.seed,
            report_to="tensorboard",  # Enable TensorBoard reporting
            logging_dir=str(args.output_dir / "tensorboard_logs_overfit"),  # Set logging directory
            logging_strategy="steps",
            logging_steps=1,  # Log loss every step for overfitting test
            dataloader_pin_memory=False,
            disable_tqdm=False,
        )

        logger.info(f"TensorBoard logs saved to: {args.output_dir / 'tensorboard_logs'}")
        logger.info("To view TensorBoard, run: tensorboard --logdir=" + str(args.output_dir / "tensorboard_logs"))

        optimizer = AdamW(overfit_model_instance.parameters(), lr=model_cfg.learning_rate)
        num_train_samples = len(train_ds)
        steps_per_epoch = math.ceil(num_train_samples / (model_cfg.batch_size * overfit_training_args.world_size))
        scheduler = OneCycleLR(
            optimizer,
            max_lr=model_cfg.learning_rate,
            epochs=model_cfg.num_epochs,
            steps_per_epoch=steps_per_epoch,
        )

        overfit_trainer = Trainer(
            model=overfit_model_instance,
            args=overfit_training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            callbacks=[TrackingCallback()],  # No EarlyStopping
            optimizers=(optimizer, scheduler),
        )

        logger.info("Trainer for overfitting initialized. Starting training...")
        overfit_trainer.train()
        logger.info("Overfitting training finished.")

        # *** After training, evaluate on the SAME small training set ***
        logger.info("** Evaluating model on the SAME small training set used for overfitting **")

        overfit_model_instance.eval()
        perfect_predictions_count = 0
        total_loss_from_model_on_train = 0.0

        # train_ds is Subset of 4 samples
        overfit_dataloader = DataLoader(train_ds, batch_size=model_cfg.batch_size)
        inspection_data = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(overfit_dataloader):
                # Ensure batch items are correctly moved to device if not already
                past_values = batch["past_values"].to(DEVICE)
                future_values_target = batch["future_values"].to(DEVICE)  # This is already -1/1
                past_observed_mask = batch["past_observed_mask"].to(DEVICE)
                future_observed_mask = batch["future_observed_mask"].to(DEVICE)  # Important for loss
                static_categorical_values = batch["static_categorical_values"].to(DEVICE)  # This is already -1/1
                freq_token = batch["freq_token"].to(DEVICE)

                # Get loss directly from the model, similar to how Trainer would
                outputs_with_loss = overfit_model_instance(
                    past_values=past_values,  # -1/1
                    past_observed_mask=past_observed_mask,
                    static_categorical_values=static_categorical_values,
                    future_values=future_values_target,  # -1/1
                    future_observed_mask=future_observed_mask,
                    freq_token=freq_token,
                )
                loss_from_model_per_batch = outputs_with_loss[0]  # MSE(logits, -1/1 targets)
                total_loss_from_model_on_train += loss_from_model_per_batch.item() * past_values.size(0)

                # For perfect prediction check, get the raw predictions separately
                outputs_for_prediction = overfit_model_instance(
                    past_values=past_values,
                    past_observed_mask=past_observed_mask,
                    static_categorical_values=static_categorical_values,
                    future_values=None,  # No labels for this call
                    future_observed_mask=None,
                    freq_token=freq_token,
                )
                predictions_raw_logits = outputs_for_prediction[0]
                predictions_tanh = torch.tanh(predictions_raw_logits)
                predictions_final_scaled = torch.where(
                    predictions_tanh > 0, torch.tensor(1.0, device=DEVICE), torch.tensor(-1.0, device=DEVICE)
                )

                # Store data for inspection (for each item in the batch)
                for i in range(past_values.shape[0]):  # Iterate through items in the current batch
                    sample_idx_in_subset = batch_idx * model_cfg.batch_size + i  # Get an overall index if needed

                    # We are interested in the parts of future_values_target and predictions_raw_logits
                    # where future_observed_mask is 1.
                    active_mask_for_sample = future_observed_mask[i].bool()  # (prediction_length, num_features)

                    # Get the actual target values where the mask is active
                    # Squeeze if num_features is 1, otherwise it's fine
                    masked_target = future_values_target[i][active_mask_for_sample]  # -1/1

                    # Get the raw logits corresponding to these active targets
                    masked_logits = predictions_raw_logits[i][active_mask_for_sample]

                    masked_tanh_preds = predictions_tanh[i][active_mask_for_sample]
                    masked_scaled_binary_preds = predictions_final_scaled[i][active_mask_for_sample]  # -1/1
                    mismatched_indices = (masked_scaled_binary_preds != masked_target).nonzero(as_tuple=True)[0]

                    # Also get the sigmoided and rounded predictions for these active parts
                    # masked_sigmoid_preds = predictions_sigmoid[i][active_mask_for_sample]
                    # masked_rounded_preds = predictions_rounded[i][active_mask_for_sample]

                    # mismatched_indices = (masked_rounded_preds != masked_target).nonzero(as_tuple=True)[0]

                    if mismatched_indices.numel() == 0:
                        perfect_predictions_count += 1
                        # Update logging to reflect -1/1 data and tanh/scaled_binary predictions
                        logger.debug(
                            f"    Mismatched scaled binary preds: {masked_scaled_binary_preds[mismatched_indices].cpu().numpy()}"
                        )
                        logger.debug(
                            f"    Mismatched targets (-1/1):       {masked_target[mismatched_indices].cpu().numpy()}"
                        )
                        logger.debug(
                            f"    Corresponding tanh:    {masked_tanh_preds[mismatched_indices].cpu().numpy().round(decimals=3)}"
                        )
                    else:
                        # This case should mean perfect_predictions_count increments, but it's not.
                        # This implies the `is_perfect_for_sample` logic itself might have an issue if this branch is hit
                        # but perfect_predictions_count doesn't go up.
                        # However, it's more likely the mismatch is real.
                        logger.debug(
                            f"  Sample {sample_idx_in_subset}: No mismatch found in this detailed check for this sample (masked parts)."
                        )
                        # If this prints, but perfect_predictions_count is 0, then the comparison logic for perfect_predictions_count is flawed.
                        # But given the previous logs, it's more likely there IS a mismatch somewhere in the full masked sequence.

                    inspection_data.append(
                        {
                            "sample_index_in_subset": sample_idx_in_subset,
                            "batch_idx": batch_idx,
                            "item_in_batch": i,
                            "raw_logits_full": predictions_raw_logits[i]
                            .cpu()
                            .numpy()
                            .tolist(),  # (prediction_length, num_features)
                            "target_full": future_values_target[i]
                            .cpu()
                            .numpy()
                            .tolist(),  # (prediction_length, num_features)
                            "future_mask_full": future_observed_mask[i]
                            .cpu()
                            .numpy()
                            .tolist(),  # (prediction_length, num_features)
                            "masked_raw_logits": masked_logits.cpu()
                            .numpy()
                            .tolist(),  # Flattened list of active logits
                            "masked_target_values": masked_target.cpu()
                            .numpy()
                            .tolist(),  # Flattened list of active targets
                            "masked_tanh_preds": masked_tanh_preds.cpu().numpy().tolist(),
                            "masked_scaled_binary_preds": masked_scaled_binary_preds.cpu().numpy().tolist(),
                            "loss_for_this_batch_from_model": loss_from_model_per_batch.item(),  # Loss for the whole batch
                        }
                    )

                    is_perfect_for_sample = mismatched_indices.numel() == 0
                    if is_perfect_for_sample:
                        perfect_predictions_count += 1

                    # # Perfect Prediction Check (on masked parts)
                    # is_perfect_for_sample = torch.all(
                    #     masked_rounded_preds == masked_target  # Comparing already masked parts
                    # )
                    # if is_perfect_for_sample:
                    #     perfect_predictions_count += 1

        avg_loss_from_model_on_train = total_loss_from_model_on_train / len(train_ds)
        logger.info(f"*** Overfitting Test Results (on the {NUM_OVERFIT_SAMPLES} training samples) ***")
        logger.info(f"Average Loss (from model's forward pass) on training data: {avg_loss_from_model_on_train:.6f}")
        logger.info(f"Number of perfectly predicted sequences (masked): {perfect_predictions_count}/{len(train_ds)}")

        # Now print the inspection_data
        logger.info("** Detailed Inspection of Predictions (First few samples) **")
        for k, data_item in enumerate(inspection_data):
            if k >= 2 and len(inspection_data) > 2:  # Print details for first 2 samples if more than 2, else print all
                logger.info(f"... (omitting details for remaining {len(inspection_data) - k} samples) ...")
                break
            logger.info(
                f"** Sample {data_item['sample_index_in_subset']} (Batch {data_item['batch_idx']}, Item {data_item['item_in_batch']}) **"
            )
            logger.info(f"  Loss for its batch (from model): {data_item['loss_for_this_batch_from_model']:.6f}")

            # To make printing manageable, let's look at a few time steps and a few features
            # for the 'full' arrays. For 'masked' arrays, they might already be small.
            num_timesteps_to_show = min(5, model_cfg.prediction_length)
            num_features_to_show = min(5, model_cfg.state_dim)

            logger.info(
                f"  Target (future_values_target) - First {num_timesteps_to_show} steps, {num_features_to_show} features:"
            )
            for t in range(num_timesteps_to_show):
                logger.info(f"    t={t}: {np.array(data_item['target_full'])[t, :num_features_to_show]}")

            logger.info(
                f"  Raw Logits (predictions_raw_logits) - First {num_timesteps_to_show} steps, {num_features_to_show} features:"
            )
            for t in range(num_timesteps_to_show):
                logger.info(
                    f"    t={t}: {np.array(data_item['raw_logits_full'])[t, :num_features_to_show].round(decimals=3)}"
                )

            logger.info(
                f"  Future Observed Mask - First {num_timesteps_to_show} steps, {num_features_to_show} features:"
            )
            for t in range(num_timesteps_to_show):
                logger.info(f"    t={t}: {np.array(data_item['future_mask_full'])[t, :num_features_to_show]}")

            # For masked values, they are already flattened. Let's show some.
            num_masked_elements_to_show = min(10, len(data_item["masked_target_values"]))
            logger.info(
                f"  Masked Target Values (first {num_masked_elements_to_show} active elements): {np.array(data_item['masked_target_values'])[:num_masked_elements_to_show]}"
            )
            logger.info(
                f"  Masked Raw Logits (first {num_masked_elements_to_show} active elements): {np.array(data_item['masked_raw_logits'])[:num_masked_elements_to_show].round(decimals=3)}"
            )
            logger.info(
                f"  Masked TanH Preds (first {num_masked_elements_to_show} active elements): {np.array(data_item['masked_tanh_preds'])[:num_masked_elements_to_show].round(decimals=3)}"
            )
            logger.info(
                f"  Masked Scaled Binary Preds (first {num_masked_elements_to_show} active elements): {np.array(data_item['masked_scaled_binary_preds'])[:num_masked_elements_to_show]}"
            )

        if avg_loss_from_model_on_train < 1e-3 and perfect_predictions_count == len(train_ds):  # Adjust threshold
            logger.success("Overfitting test PASSED: Model successfully memorized the small batch.")
        else:
            logger.error("Overfitting test FAILED: Model did not memorize the small batch.")
            logger.error(
                "Check data processing, `__getitem__`, loss function alignment, and model input/output shapes. "
                "Inspect the printed raw logits and targets above."
            )

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
    main_overfit()
