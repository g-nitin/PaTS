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
from torch.utils.tensorboard.writer import SummaryWriter
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer import Trainer
from transformers.trainer_callback import EarlyStoppingCallback, TrainerCallback
from transformers.trainer_utils import set_seed
from transformers.training_args import TrainingArguments
from tsfm_public import TrackingCallback  # type: ignore
from tsfm_public.toolkit.get_model import get_model  # type: ignore

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
            optimizers=(optimizer, scheduler),  # type: ignore
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


def main():
    parser = argparse.ArgumentParser(description="Train a TTM model on the BlocksWorld domain.")
    parser.add_argument("--dataset_file", type=Path, required=True, help="Path to the JSON dataset file.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output_blocksworld_ttm"),
        help="Directory to save models, logs, and results.",
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
    log_file = args.output_dir / "run_ttm.log"
    setup_logging(args.log_level, log_file)

    logger.info(f"Script arguments: {pformat(vars(args))}")

    # Set seed
    set_seed(args.seed)
    logger.info(f"Random seed set to: {args.seed}")

    # Infer the number of blocks
    num_blocks = int(str(args.dataset_file).split("_")[-1][0])
    logger.info(f"Inferred number of blocks from dataset filename: {num_blocks}")

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

    train_ds, val_ds, _ = prepare_datasets(
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

    logger.info(f"Training complete. Model saved to {final_model_assets_path}")
    logger.info("To evaluate, use the benchmark.py script.")

    logger.info("TTM script finished.")
    return 0


if __name__ == "__main__":
    main()
