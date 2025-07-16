import json
import math
import posixpath
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path, PosixPath
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from loguru import logger
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset  # Used for type hinting, PaTSDataset will be the actual one
from torch.utils.tensorboard.writer import SummaryWriter
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer import Trainer
from transformers.trainer_callback import EarlyStoppingCallback, TrainerCallback
from transformers.training_args import TrainingArguments
from tsfm_public import TrackingCallback  # type: ignore
from tsfm_public.toolkit.get_model import get_model  # type: ignore

from ..pats_dataset import PaTSDataset

# ** Constants **
DEFAULT_TTM_MODEL_PATH = "ibm-granite/granite-timeseries-ttm-r2.1"  # Default TTM model
# Define the supported models based on the provided combinations: https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2/tree/main
SUPPORTED_MODELS = [
    {"context_length": 52, "prediction_length": 16, "freq_tuning": True, "loss_metric": "mse", "release": "r2.1"},
    {"context_length": 52, "prediction_length": 16, "freq_tuning": True, "loss_metric": "mae", "release": "r2.1"},
    {"context_length": 90, "prediction_length": 30, "freq_tuning": True, "loss_metric": "mse", "release": "r2.1"},
    {"context_length": 90, "prediction_length": 30, "freq_tuning": True, "loss_metric": "mae", "release": "r2.1"},
    {"context_length": 180, "prediction_length": 60, "freq_tuning": True, "loss_metric": "mae", "release": "r2.1"},
    {"context_length": 360, "prediction_length": 60, "freq_tuning": True, "loss_metric": "mae", "release": "r2.1"},
    {"context_length": 1024, "prediction_length": 96, "freq_tuning": False, "loss_metric": "mse", "release": "r2"},
    {"context_length": 1024, "prediction_length": 192, "freq_tuning": False, "loss_metric": "mse", "release": "r2"},
    {"context_length": 1024, "prediction_length": 336, "freq_tuning": False, "loss_metric": "mse", "release": "r2"},
    {"context_length": 1024, "prediction_length": 720, "freq_tuning": False, "loss_metric": "mse", "release": "r2"},
    {"context_length": 512, "prediction_length": 48, "freq_tuning": True, "loss_metric": "mse", "release": "r2.1"},
    {"context_length": 512, "prediction_length": 48, "freq_tuning": True, "loss_metric": "mae", "release": "r2.1"},
    {"context_length": 512, "prediction_length": 96, "freq_tuning": True, "loss_metric": "mse", "release": "r2.1"},
    {"context_length": 512, "prediction_length": 96, "freq_tuning": True, "loss_metric": "mae", "release": "r2.1"},
    {"context_length": 512, "prediction_length": 192, "freq_tuning": False, "loss_metric": "mse", "release": "r2"},
    {"context_length": 512, "prediction_length": 336, "freq_tuning": False, "loss_metric": "mse", "release": "r2"},
    {"context_length": 512, "prediction_length": 720, "freq_tuning": False, "loss_metric": "mse", "release": "r2"},
    {"context_length": 1536, "prediction_length": 96, "freq_tuning": False, "loss_metric": "mse", "release": "r2"},
    {"context_length": 1536, "prediction_length": 192, "freq_tuning": False, "loss_metric": "mse", "release": "r2"},
    {"context_length": 1536, "prediction_length": 336, "freq_tuning": False, "loss_metric": "mse", "release": "r2"},
    {"context_length": 1536, "prediction_length": 720, "freq_tuning": False, "loss_metric": "mse", "release": "r2"},
]

global DEVICE
DEVICE: torch.device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
# DEVICE = torch.device("cpu")


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


class TTMDataCollator:
    """
    Data collator for TTM model.
    Takes a list of dictionaries from PaTSDataset (each containing 'initial_state', 'goal_state', 'expert_trajectory', 'id' as numpy arrays) and transforms them into a batch dictionary suitable for TTM input.
    This includes creating past/future values, masks, scaling, and padding.
    """

    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        state_dim: int,  # state_dim is crucial for creating correct shapes
    ):
        self.context_length: int = context_length
        self.prediction_length: int = prediction_length
        self.state_dim: int = state_dim

    def _scale_binary_array(self, data_array_np: np.ndarray) -> np.ndarray:
        """Scales a numpy array of 0s and 1s to -1s and 1s."""
        return data_array_np * 2.0 - 1.0

    def __call__(self, batch_items: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
        # batch_items is a list of dicts from PaTSDataset.__getitem__
        # Each dict: {'initial_state', 'goal_state', 'expert_trajectory', 'id'}
        # All values are np.float32 numpy arrays.

        batch_size = len(batch_items)

        list_past_values_scaled = []
        list_future_values_scaled = []
        list_past_observed_mask = []
        list_future_observed_mask = []
        list_static_categorical_scaled = []

        # print(batch_items)
        for item in batch_items:
            plan_states_np_orig = item["expert_trajectory"]  # (L, F), 0/1
            goal_state_np_orig = item["goal_state"]  # (F,), 0/1
            initial_state_np_orig = item["initial_state"]  # (F,), 0/1

            past_values_np = np.zeros((self.context_length, self.state_dim), dtype=np.float32)
            past_observed_mask_np = np.zeros((self.context_length, self.state_dim), dtype=np.float32)

            num_plan_steps_for_context = min(len(plan_states_np_orig), self.context_length)

            if num_plan_steps_for_context > 0:
                past_values_np[:num_plan_steps_for_context] = plan_states_np_orig[:num_plan_steps_for_context]
                past_observed_mask_np[:num_plan_steps_for_context, :] = 1.0
                if num_plan_steps_for_context < self.context_length:
                    last_observed_state_in_context = plan_states_np_orig[num_plan_steps_for_context - 1]
                    num_past_padding = self.context_length - num_plan_steps_for_context
                    padding_values = np.tile(last_observed_state_in_context, (num_past_padding, 1))
                    past_values_np[num_plan_steps_for_context:] = padding_values
            elif self.context_length > 0:  # plan_states_np_orig might be empty or shorter than context
                padding_values = np.tile(initial_state_np_orig, (self.context_length, 1))
                past_values_np[:] = padding_values

            future_values_np = np.zeros((self.prediction_length, self.state_dim), dtype=np.float32)
            future_observed_mask_np = np.zeros((self.prediction_length, self.state_dim), dtype=np.float32)

            target_future_plan_states_orig_list = []
            if len(plan_states_np_orig) > self.context_length:
                target_future_plan_states_orig_list = plan_states_np_orig[self.context_length :]

            target_future_plan_states_orig = np.array(target_future_plan_states_orig_list, dtype=np.float32)
            num_actual_future_steps_from_plan = (
                target_future_plan_states_orig.shape[0] if target_future_plan_states_orig.ndim == 2 else 0
            )
            len_to_copy_to_future = min(num_actual_future_steps_from_plan, self.prediction_length)

            if len_to_copy_to_future > 0:
                future_values_np[:len_to_copy_to_future] = target_future_plan_states_orig[:len_to_copy_to_future]
                future_observed_mask_np[:len_to_copy_to_future, :] = 1.0

            if len_to_copy_to_future < self.prediction_length:
                num_future_padding = self.prediction_length - len_to_copy_to_future
                padding_values = np.tile(goal_state_np_orig, (num_future_padding, 1))
                future_values_np[len_to_copy_to_future:] = padding_values
                future_observed_mask_np[len_to_copy_to_future:, :] = 1.0

            static_categorical_values_np = goal_state_np_orig.copy()

            list_past_values_scaled.append(self._scale_binary_array(past_values_np))
            list_future_values_scaled.append(self._scale_binary_array(future_values_np))
            list_past_observed_mask.append(past_observed_mask_np)
            list_future_observed_mask.append(future_observed_mask_np)
            list_static_categorical_scaled.append(self._scale_binary_array(static_categorical_values_np))

        # Stack and convert to tensors. Trainer will move to device.
        return {
            "freq_token": torch.zeros(batch_size, dtype=torch.long),
            "past_values": torch.from_numpy(np.array(list_past_values_scaled, dtype=np.float32)),
            "future_values": torch.from_numpy(np.array(list_future_values_scaled, dtype=np.float32)),
            "past_observed_mask": torch.from_numpy(np.array(list_past_observed_mask, dtype=np.float32)),
            "future_observed_mask": torch.from_numpy(np.array(list_future_observed_mask, dtype=np.float32)),
            "static_categorical_values": torch.from_numpy(np.array(list_static_categorical_scaled, dtype=np.float32)),
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
        logger.info(f"Initializing TTM model from: {self.config.ttm_model_path}")

        get_model_params = {
            "model_path": self.config.ttm_model_path,
            "context_length": self.config.context_length,
            "prediction_length": self.config.prediction_length,
            "head_dropout": 0.1,
        }
        self.model: PreTrainedModel = get_model(**get_model_params).to(self.device)  # type: ignore

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
            remove_unused_columns=False,  # Do not remove unused columns
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

        # Instantiate the data collator
        ttm_collator = TTMDataCollator(
            context_length=self.config.context_length,
            prediction_length=self.config.prediction_length,
            state_dim=self.config.state_dim,  # type: ignore
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=callbacks,
            data_collator=ttm_collator,  # Use the custom collator
            optimizers=(optimizer, scheduler),  # type: ignore
        )

        logger.info("Trainer initialized. Starting training...")
        self.trainer.train()
        logger.info("Training finished.")

        # Close TensorBoard writer
        if self.tb_writer:
            self.tb_writer.close()

    def predict(self, context_sequence: torch.Tensor, goal_states: torch.Tensor) -> torch.Tensor:
        """
        Predicts a sequence of future states given a context sequence and goal states.
        This is the core model call used by the autoregressive loop.

        :param context_sequence: A tensor of shape (batch_size, context_length, num_features) containing the recent history of states (in 0/1 format).
        :param goal_states: A tensor of shape (batch_size, num_features) for the goal states (in 0/1 format).
        :return: A tensor of shape (batch_size, prediction_length, num_features) containing the predicted future states (in 0/1 format).
        """
        if self.model is None:
            raise RuntimeError("Model needs to be trained or loaded before prediction.")
        if self.config.state_dim is None:
            raise RuntimeError("Model config state_dim is not set.")

        self.model.eval()
        with torch.no_grad():
            batch_size = context_sequence.shape[0]

            # Scale inputs from 0/1 to -1/1 for the model
            context_sequence_scaled = context_sequence.to(self.device) * 2.0 - 1.0
            goal_states_scaled = goal_states.to(self.device) * 2.0 - 1.0

            inputs = {
                "past_values": context_sequence_scaled,
                "past_observed_mask": torch.ones_like(context_sequence_scaled).to(self.device),
                "static_categorical_values": goal_states_scaled,
                "freq_token": torch.zeros(batch_size, dtype=torch.long).to(self.device),
            }

            outputs = self.model(**inputs)
            raw_logits = outputs[0]  # Shape: (batch_size, prediction_length, num_features)

            # Binarize the output logits
            predictions_tanh = torch.tanh(raw_logits)
            predictions_scaled_binary = torch.where(
                predictions_tanh > 0, torch.tensor(1.0, device=self.device), torch.tensor(-1.0, device=self.device)
            )

            # Convert back from -1/1 to 0/1 for the return value
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
        instance.model = get_model(**get_model_params).to(instance.device)  # type: ignore

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


def load_problem_basenames_from_split_file(split_file_path: Path) -> List[str]:
    if not split_file_path.exists():
        logger.warning(f"Split file not found: {split_file_path}")
        return []
    with open(split_file_path, "r") as f:
        basenames = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return basenames


def get_model_path(model_config: Dict[str, Union[int, bool, str]]) -> str:
    """Constructs the model path from its configuration."""
    parts = ["ibm-granite/granite-timeseries-ttm-r2"]  # Base path
    config_str = f"{model_config['context_length']}-{model_config['prediction_length']}"
    if model_config["freq_tuning"]:
        config_str += "-ft"
    if model_config["loss_metric"] == "mae":
        # Note: Using 'l1' as it appears in the user-provided list.
        config_str += "-l1"
    config_str += f"-{model_config['release']}"
    return f"{parts[0]}/{config_str}"


def determine_ttm_model(
    max_plan_length: int,
    recommended_prediction_length: int,
    user_context_length: Optional[int] = None,
    user_prediction_length: Optional[int] = None,
    user_freq_tuning: Optional[bool] = None,
    user_loss_metric: Optional[str] = None,
    user_release: Optional[str] = None,
) -> Optional[Dict[str, Union[int, bool, str]]]:
    """
    Determines the optimal TTM model based on constraints and user preferences.
    """
    candidates = SUPPORTED_MODELS

    # 1. Filter based on user overrides
    if user_context_length is not None:
        candidates = [m for m in candidates if m["context_length"] == user_context_length]
    if user_prediction_length is not None:
        candidates = [m for m in candidates if m["prediction_length"] == user_prediction_length]
    if user_freq_tuning is not None:
        candidates = [m for m in candidates if m["freq_tuning"] == user_freq_tuning]
    if user_loss_metric is not None:
        candidates = [m for m in candidates if m["loss_metric"] == user_loss_metric]
    if user_release is not None:
        candidates = [m for m in candidates if m["release"] == user_release]

    if not candidates:
        logger.warning("No models match the specified user overrides.")
        return None

    # 2. Filter by max_plan_length
    valid_candidates = [m for m in candidates if m["context_length"] <= max_plan_length]
    if not valid_candidates:
        min_supported_cl = min(m["context_length"] for m in candidates)
        logger.warning(
            f"Max plan length ({max_plan_length}) is smaller than all supported context lengths for the current selection. "
            f"Considering models with the smallest supported context length: {min_supported_cl}."
        )
        # Relax the constraint to the smallest possible context length among the candidates
        valid_candidates = [m for m in candidates if m["context_length"] == min_supported_cl]

    # 3. Find the best prediction_length
    # Find models with prediction length <= recommended
    pl_candidates = [m for m in valid_candidates if m["prediction_length"] <= recommended_prediction_length]

    if pl_candidates:
        # If there are such models, find the largest prediction length among them
        best_pl = max(m["prediction_length"] for m in pl_candidates)
        final_candidates = [m for m in pl_candidates if m["prediction_length"] == best_pl]
    else:
        # Otherwise, choose the smallest available prediction length
        best_pl = min(m["prediction_length"] for m in valid_candidates)
        logger.warning(
            f"Recommended prediction length ({recommended_prediction_length}) is smaller than all supported forecast lengths. "
            f"Choosing the smallest supported: {best_pl}."
        )
        final_candidates = [m for m in valid_candidates if m["prediction_length"] == best_pl]

    logger.info(
        f"Recommended prediction length: {recommended_prediction_length} | Auto-selected prediction_length: {best_pl}"
    )

    # 4. Select the one with the largest context_length from the finalists
    best_cl = max(m["context_length"] for m in final_candidates)
    final_candidates = [m for m in final_candidates if m["context_length"] == best_cl]
    logger.info(f"Max plan length: {max_plan_length} | Auto-selected context_length: {best_cl}")

    # 5. Tie-break if multiple models remain
    # Sort by: freq_tuning (True first), loss_metric ('mae' first), release (e.g., 'r2.1' > 'r2')
    final_candidates.sort(key=lambda m: (not m["freq_tuning"], m["loss_metric"] != "mae", m["release"]), reverse=True)

    selected_model = final_candidates[0]

    logger.info(f"Selected model configuration: {selected_model}")
    logger.info(f"Model path: {get_model_path(selected_model)}")

    return selected_model


def get_num_blocks_from_filename(filename_base: str) -> Optional[int]:
    """Extracts number of blocks from a filename like 'blocks_3_problem_1'."""
    match = re.search(r"blocks_(\d+)_problem", filename_base)
    if match:
        return int(match.group(1))
    return None


def prepare_datasets(
    dataset_dir: Path, dataset_split_dir: Path, num_blocks: int, context_length: int, prediction_length: int, seed: int
) -> Tuple[Dataset, Dataset, Dataset, int, int]:
    logger.info("Preparing train/validation/test datasets...")

    train_basenames_all = load_problem_basenames_from_split_file(dataset_split_dir / "train_files.txt")
    val_basenames_all = load_problem_basenames_from_split_file(dataset_split_dir / "val_files.txt")
    # Test dataset is not strictly needed for training, but can be prepared for completeness
    # test_basenames_all = load_problem_basenames_from_split_file(dataset_split_dir / "test_files.txt")

    # Filter basenames for the target num_blocks (important if split files are mixed)
    # Assuming dataset_dir is already blocks_<N> specific, this might be redundant but safe.
    train_basenames = [bn for bn in train_basenames_all if get_num_blocks_from_filename(bn) == num_blocks]
    val_basenames = [bn for bn in val_basenames_all if get_num_blocks_from_filename(bn) == num_blocks]
    # test_basenames = [bn for bn in test_basenames_all if get_num_blocks_from_filename(bn) == num_blocks] # Test not strictly needed for training

    if not train_basenames:
        raise ValueError(
            f"No training problem basenames found for N={num_blocks} in {dataset_split_dir / 'train_files.txt'}"
        )

    logger.info(f"Found {len(train_basenames)} train, {len(val_basenames)} val basenames for N={num_blocks}.")

    # Create dataset instances
    # PaTSDataset will infer state_dim from the data.
    # dataset_dir is e.g. data/blocks_4, split_file_name is e.g. "train_files.txt"
    train_dataset_instance = PaTSDataset(dataset_dir=dataset_dir, split_file_name="train_files.txt")
    state_dim = train_dataset_instance.state_dim
    val_dataset_instance = PaTSDataset(dataset_dir=dataset_dir, split_file_name="val_files.txt")
    test_dataset_instance = PaTSDataset(dataset_dir=dataset_dir, split_file_name="test_files.txt")

    # Get max_plan_len_in_train_data using PaTSDataset structure
    max_plan_len_in_train_data = 0
    if len(train_dataset_instance.basenames) > 0:
        for basename_for_len_check in train_dataset_instance.basenames:
            traj_file_path_for_len = train_dataset_instance.trajectories_bin_dir / f"{basename_for_len_check}.traj.bin.npy"
            if traj_file_path_for_len.exists():
                try:
                    traj_np = np.load(traj_file_path_for_len)
                    if traj_np.ndim == 2:
                        max_plan_len_in_train_data = max(max_plan_len_in_train_data, traj_np.shape[0])
                except Exception as e:
                    logger.warning(f"Could not load trajectory {traj_file_path_for_len} to determine max length: {e}")
            else:
                logger.warning(f"Trajectory file {traj_file_path_for_len} not found during max length check.")

    logger.info(
        f"Dataset preparation complete. state_dim: {state_dim}, max_plan_len_in_train_data: {max_plan_len_in_train_data}"
    )

    return train_dataset_instance, val_dataset_instance, test_dataset_instance, state_dim, max_plan_len_in_train_data
