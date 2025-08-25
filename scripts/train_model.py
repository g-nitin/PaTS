import argparse
import sys
import warnings
from datetime import datetime
from functools import partial
from pathlib import Path
from pprint import pprint
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer import Trainer
from transformers.trainer_utils import set_seed as ttm_set_seed
from transformers.training_args import TrainingArguments

from scripts.BlocksWorldValidator import BlocksWorldValidator
from scripts.models.llama import LlamaFinetuneDataCollator, LlamaFinetuneDataset
from scripts.models.lstm import PaTS_LSTM, lstm_collate_fn
from scripts.models.ttm import BlocksWorldTTM, determine_ttm_model
from scripts.models.ttm import ModelConfig as TTMModelConfig
from scripts.models.ttm import setup_logging as ttm_setup_logging
from scripts.models.xgboost import XGBoostPlanner
from scripts.pats_dataset import PaTSDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))


def train_lstm_model_loop(model, train_loader, val_loader, validator, args, num_features, model_save_path):
    print("Starting LSTM training...")

    # Select Loss Function based on Encoding
    if args.encoding_type == "sas":
        # For SAS+, the model outputs logits for each possible location (class) for each block.
        # The target is the index of the correct location.
        criterion = nn.CrossEntropyLoss()
        print("Using CrossEntropyLoss for SAS+ encoding.")
    else:  # binary
        criterion = nn.BCEWithLogitsLoss(reduction="none")
        print("Using BCEWithLogitsLoss for binary encoding.")

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=10, factor=0.5)

    best_val_loss = float("inf")
    sas_clamp_warning_issued = False

    for epoch in range(args.epochs):
        model.train()
        epoch_train_loss, epoch_forecast_loss, epoch_mlm_loss, epoch_constraint_loss = 0.0, 0.0, 0.0, 0.0
        num_train_batches = 0

        for batch_data in train_loader:
            if batch_data is None:
                continue
            input_seqs = batch_data["input_sequences"].to(DEVICE)
            goal_states = batch_data["goal_states"].to(DEVICE)
            target_seqs = batch_data["target_sequences"].to(DEVICE)
            lengths = batch_data["lengths"]
            # MLM mask is only relevant for binary encoding
            mlm_predicate_mask = batch_data.get("mlm_predicate_mask", torch.tensor([])).to(DEVICE)

            # Safeguard for SAS+ encoding to prevent out-of-bounds embedding errors.
            if args.encoding_type == "sas":
                max_val = args.num_blocks
                # Check if any values are out of the expected [0, max_val] range.
                if not sas_clamp_warning_issued and (torch.any(input_seqs > max_val) or torch.any(goal_states > max_val)):
                    warnings.warn(
                        f"SAS+ input data contains values > num_blocks ({max_val}). "
                        f"Clamping values to prevent embedding layer errors. "
                        f"Please check your dataset for correctness."
                    )
                    sas_clamp_warning_issued = True
                input_seqs.clamp_(min=0, max=max_val)
                goal_states.clamp_(min=0, max=max_val)

            optimizer.zero_grad()
            # For SAS+, mlm_logits will be None
            forecasting_logits, mlm_logits, _ = model(input_seqs, goal_states, lengths)

            # Modified Loss Calculation
            if args.encoding_type == "sas":
                # For CrossEntropyLoss, logits should be (N, C) and targets (N)
                # N = total number of blocks to predict across batch, C = num_locations
                # Create a mask to select only the valid time steps based on sequence lengths
                mask = (
                    torch.arange(max(lengths), device=DEVICE)[None, :] < lengths.clone().detach().to(DEVICE)[:, None]
                )  # (B, S_max)

                # Reshape logits and targets and apply the mask
                # Logits: (B, S_max, N_blocks, N_locs) -> (num_active_steps, N_blocks, N_locs)
                active_logits = forecasting_logits[mask]
                # Targets: (B, S_max, N_blocks) -> (num_active_steps, N_blocks)
                active_targets = target_seqs[mask]

                if active_targets.numel() == 0:
                    continue  # Skip batch if no valid targets

                # Map SAS+ target values (e.g., 0 for table, 1..N for blocks) to class indices
                active_targets_indices = model._map_sas_to_indices(active_targets)

                # Final reshape for loss function
                # (num_active_steps * N_blocks, N_locs) and (num_active_steps * N_blocks)
                loss_forecasting = criterion(
                    active_logits.reshape(-1, model.num_locations), active_targets_indices.reshape(-1)
                )
            else:  # Binary encoding loss calculation
                forecasting_mask = torch.zeros_like(target_seqs, dtype=torch.bool).to(DEVICE)
                for i, length_val in enumerate(lengths):
                    if length_val > 0:
                        forecasting_mask[i, :length_val, :] = True

                num_forecast_elements = forecasting_mask.float().sum()
                if num_forecast_elements == 0:
                    continue

                loss_forecasting_unreduced = criterion(forecasting_logits, target_seqs)
                loss_forecasting = (loss_forecasting_unreduced * forecasting_mask.float()).sum() / num_forecast_elements

            # For SAS+, MLM and Constraint losses are not used. They will be zero.
            loss_mlm = torch.tensor(0.0).to(DEVICE)
            if args.encoding_type != "sas" and args.use_mlm_task and mlm_logits is not None:
                num_masked_elements = mlm_predicate_mask.sum()
                if num_masked_elements > 0:
                    loss_mlm_unreduced = criterion(mlm_logits, input_seqs)
                    loss_mlm = (loss_mlm_unreduced * mlm_predicate_mask).sum() / num_masked_elements

            loss_constraint = torch.tensor(0.0).to(DEVICE)
            if args.encoding_type != "sas" and args.use_constraint_loss and validator is not None:
                masked_logits = forecasting_logits[forecasting_mask[:, :, 0]]  # type: ignore
                if masked_logits.ndim == 2 and masked_logits.shape[0] > 0:
                    loss_constraint = validator.calculate_constraint_violation_loss(masked_logits)

            # Total Loss
            total_loss = loss_forecasting + args.mlm_loss_weight * loss_mlm + args.constraint_loss_weight * loss_constraint
            total_loss.backward()
            if args.clip_grad_norm is not None and args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

            epoch_train_loss += total_loss.item()
            epoch_forecast_loss += loss_forecasting.item()
            epoch_mlm_loss += loss_mlm.item()
            epoch_constraint_loss += loss_constraint.item()
            num_train_batches += 1

        avg_train_loss = epoch_train_loss / num_train_batches if num_train_batches > 0 else float("inf")
        avg_forecast_loss = epoch_forecast_loss / num_train_batches if num_train_batches > 0 else float("inf")
        avg_mlm_loss = epoch_mlm_loss / num_train_batches if num_train_batches > 0 else float("inf")
        avg_constraint_loss = epoch_constraint_loss / num_train_batches if num_train_batches > 0 else float("inf")

        # Validation
        model.eval()
        epoch_val_loss, epoch_val_forecast_loss, epoch_val_mlm_loss, epoch_val_constraint_loss = 0.0, 0.0, 0.0, 0.0
        num_val_batches = 0
        with torch.no_grad():
            if val_loader is not None:
                for batch_data in val_loader:
                    if batch_data is None:
                        continue
                    input_seqs = batch_data["input_sequences"].to(DEVICE)
                    goal_states = batch_data["goal_states"].to(DEVICE)
                    target_seqs = batch_data["target_sequences"].to(DEVICE)
                    lengths = batch_data["lengths"]
                    mlm_predicate_mask = batch_data.get("mlm_predicate_mask", torch.tensor([])).to(DEVICE)

                    if args.encoding_type == "sas":
                        input_seqs.clamp_(min=0, max=args.num_blocks)
                        goal_states.clamp_(min=0, max=args.num_blocks)

                    forecasting_logits, mlm_logits, _ = model(input_seqs, goal_states, lengths)

                    if args.encoding_type == "sas":
                        mask = (
                            torch.arange(max(lengths), device=DEVICE)[None, :]
                            < lengths.clone().detach().to(DEVICE)[:, None]
                        )
                        active_logits = forecasting_logits[mask]
                        active_targets = target_seqs[mask]

                        if active_targets.numel() == 0:
                            continue

                        active_targets_indices = model._map_sas_to_indices(active_targets)
                        loss_forecasting = criterion(
                            active_logits.reshape(-1, model.num_locations), active_targets_indices.reshape(-1)
                        )
                    else:  # Binary
                        forecasting_mask = torch.zeros_like(target_seqs, dtype=torch.bool).to(DEVICE)
                        for i, length_val in enumerate(lengths):
                            if length_val > 0:
                                forecasting_mask[i, :length_val, :] = True
                        num_forecast_elements = forecasting_mask.float().sum()
                        if num_forecast_elements == 0:
                            continue
                        loss_forecasting_unreduced = criterion(forecasting_logits, target_seqs)
                        loss_forecasting = (
                            loss_forecasting_unreduced * forecasting_mask.float()
                        ).sum() / num_forecast_elements

                    loss_mlm = torch.tensor(0.0).to(DEVICE)
                    if args.encoding_type != "sas" and args.use_mlm_task and mlm_logits is not None:
                        num_masked_elements = mlm_predicate_mask.sum()
                        if num_masked_elements > 0:
                            loss_mlm_unreduced = criterion(mlm_logits, input_seqs)
                            loss_mlm = (loss_mlm_unreduced * mlm_predicate_mask).sum() / num_masked_elements

                    loss_constraint = torch.tensor(0.0).to(DEVICE)
                    if args.encoding_type != "sas" and args.use_constraint_loss and validator is not None:
                        masked_logits = forecasting_logits[forecasting_mask[:, :, 0]]  # type: ignore
                        if masked_logits.ndim == 2 and masked_logits.shape[0] > 0:
                            loss_constraint = validator.calculate_constraint_violation_loss(masked_logits)

                    total_loss = (
                        loss_forecasting + args.mlm_loss_weight * loss_mlm + args.constraint_loss_weight * loss_constraint
                    )
                    epoch_val_loss += total_loss.item()
                    epoch_val_forecast_loss += loss_forecasting.item()
                    epoch_val_mlm_loss += loss_mlm.item()
                    epoch_val_constraint_loss += loss_constraint.item()
                    num_val_batches += 1

        avg_val_loss = epoch_val_loss / num_val_batches if num_val_batches > 0 else float("inf")
        avg_val_forecast_loss = epoch_val_forecast_loss / num_val_batches if num_val_batches > 0 else float("inf")
        avg_val_mlm_loss = epoch_val_mlm_loss / num_val_batches if num_val_batches > 0 else float("inf")
        avg_val_constraint_loss = epoch_val_constraint_loss / num_val_batches if num_val_batches > 0 else float("inf")
        scheduler.step(avg_val_loss)

        train_loss_str = f"Train Loss: {avg_train_loss:.4f} (F: {avg_forecast_loss:.4f}, M: {avg_mlm_loss:.4f}, C: {avg_constraint_loss:.4f})"
        val_loss_str = f"Val Loss: {avg_val_loss:.4f} (F: {avg_val_forecast_loss:.4f}, M: {avg_val_mlm_loss:.4f}, C: {avg_val_constraint_loss:.4f})"
        print(f"Epoch [{epoch + 1}/{args.epochs}] {train_loss_str}, {val_loss_str}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_val_loss,
                    "encoding_type": args.encoding_type,
                    "num_features": num_features,
                    "hidden_size": args.lstm_hidden_size,
                    "num_lstm_layers": args.lstm_num_layers,
                    "dropout_prob": args.lstm_dropout_prob,
                    "use_mlm_task": args.use_mlm_task,
                    "mlm_loss_weight": args.mlm_loss_weight,
                    "target_num_blocks": args.num_blocks,
                    "embedding_dim": args.lstm_embedding_dim if args.encoding_type == "sas" else None,
                },
                model_save_path,
            )
            print(f"Model saved to {model_save_path} (Val Loss: {best_val_loss:.4f})")
    print("LSTM Training finished.")


def prepare_data_for_xgboost(dataset: PaTSDataset, context_window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flattens the sequential dataset into a tabular format (X, y) for XGBoost,
    including a context window of past states.
    X = (S_{t-k+1}, ..., S_t, S_G)
    y = S_{t+1}
    """
    X_list, y_list = [], []
    for i in range(len(dataset)):
        item = dataset[i]
        trajectory = item["expert_trajectory"]  # (L, F)
        goal_state = item["goal_state"]  # (F,)
        initial_state = item["initial_state"]  # (F,)

        # Create (S_{t-k+1}, ..., S_t, S_G) -> S_{t+1} pairs
        for t in range(len(trajectory) - 1):  # Iterate from S_0 to S_{L-2} to predict S_1 to S_{L-1}
            current_state = trajectory[t]
            next_state = trajectory[t + 1]

            # Build the context window for S_t
            context_states = []
            for j in range(context_window_size):
                idx_in_traj = t - (context_window_size - 1 - j)
                if idx_in_traj >= 0:
                    context_states.append(trajectory[idx_in_traj])
                else:
                    # Pad with initial_state if not enough history
                    context_states.append(initial_state)

            # Concatenate context states and goal state to form the feature vector X
            X_sample = np.concatenate(context_states + [goal_state])
            X_list.append(X_sample)
            y_list.append(next_state)

    return np.array(X_list), np.array(y_list)


def main():
    print("Starting unified training script for PaTS models...")

    parser = argparse.ArgumentParser(description="Unified Training Script for PaTS Models")

    # Common arguments
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["lstm", "ttm", "xgboost", "llama", "llama_finetune"],
        help="Type of model to train.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        required=True,
        help="Path to the PaTS dataset directory for a specific N (e.g., data/blocks_4).",
    )
    parser.add_argument(
        "--dataset_split_dir",
        type=Path,
        required=True,
        help="Path to the directory containing train_files.txt, etc. (e.g., data/blocks_4).",
    )
    parser.add_argument("--num_blocks", type=int, required=True, help="Number of blocks for this training run.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Base directory to save trained models and logs.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--encoding_type",
        type=str,
        default="bin",
        choices=["bin", "sas"],
        help="The encoding type of the dataset to use.",
    )

    # LSTM specific arguments
    parser.add_argument("--lstm_hidden_size", type=int, default=128, help="Hidden size for LSTM.")
    parser.add_argument("--lstm_num_layers", type=int, default=2, help="Number of LSTM layers.")
    parser.add_argument("--lstm_dropout_prob", type=float, default=0.2, help="Dropout probability for LSTM.")
    parser.add_argument(
        "--clip_grad_norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm for LSTM (0 to disable). TTM handles internally.",
    )
    parser.add_argument("--use_mlm_task", action="store_true", help="Enable MLM auxiliary task for LSTM.")
    parser.add_argument("--mlm_loss_weight", type=float, default=0.2, help="Weight for the MLM auxiliary loss.")
    parser.add_argument(
        "--mlm_mask_prob", type=float, default=0.15, help="Probability of masking a predicate for the MLM task."
    )
    parser.add_argument("--lstm_embedding_dim", type=int, default=32, help="Embedding dimension for SAS+ encoding.")
    parser.add_argument(
        "--use_constraint_loss", action="store_true", help="Enable constraint violation auxiliary loss for LSTM."
    )
    parser.add_argument(
        "--constraint_loss_weight", type=float, default=1.0, help="Weight for the constraint violation auxiliary loss."
    )

    # TTM specific arguments
    parser.add_argument(
        "--ttm_model_path",
        type=str,
        default="ibm-granite/granite-timeseries-ttm-r2",
        help="Base TTM model path from HuggingFace or local.",
    )
    parser.add_argument(
        "--context_length", type=int, help="Context length for TTM. If not provided, determined from dataset."
    )
    parser.add_argument(
        "--prediction_length", type=int, help="Prediction length for TTM. If not provided, determined from dataset."
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level for TTM.",
    )

    # XGBoost specific arguments
    parser.add_argument(
        "--xgboost_context_window_size",
        type=int,
        default=1,
        help="Number of past states to include in XGBoost input features. Default is 1 (current state only).",
    )

    # Llama fine-tuning specific arguments
    parser.add_argument(
        "--llama_model_id",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Base Llama model ID for fine-tuning.",
    )
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA attention dimension.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="Alpha parameter for LoRA scaling.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="Dropout probability for LoRA layers.")
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated list of module names to apply LoRA to.",
    )
    parser.add_argument("--llama_max_seq_len", type=int, default=512, help="Maximum sequence length for Llama fine-tuning.")
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate gradients for."
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0.03, help="Ratio of total training steps for a linear warmup."
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every N steps.")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every N steps.")
    parser.add_argument("--eval_steps", type=int, default=100, help="Run evaluation every N steps.")
    parser.add_argument(
        "--llama_use_few_shot",
        action="store_true",
        help="For Llama models, include a one-shot example in the prompt for few-shot inference.",
    )

    args = parser.parse_args()

    print("Parsed Arguments:")
    pprint(vars(args))

    if args.model_type == "ttm" and (args.mlm_loss_weight or args.mlm_mask_prob):
        warnings.warn("MLM-related arguments are not applicable for TTM. Ignoring them.")

    if args.model_type == "ttm" and (args.use_constraint_loss or args.constraint_loss_weight != 1.0):
        warnings.warn("Constraint loss arguments are not applicable for TTM. Ignoring them.")

    print(f"Using device: {DEVICE}")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.model_type == "ttm":
        ttm_set_seed(args.seed)  # Specific seed setting for TTM/HuggingFace Trainer

    # Create model-specific output directory: <output_dir>/<model_type>_N<num_blocks>/
    model_specific_output_dir = args.output_dir / f"{args.model_type}_N{args.num_blocks}"
    model_specific_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model outputs will be saved to: {model_specific_output_dir}")

    # Instantiate the validator for use in training
    validator = None
    if args.model_type == "lstm" and args.use_constraint_loss:
        print("Constraint loss enabled. Initializing BlocksWorldValidator...")
        if args.encoding_type == "bin":
            manifest_path = args.dataset_dir / f"predicate_manifest_{args.num_blocks}.txt"
            if not manifest_path.exists():
                print(
                    f"ERROR: Predicate manifest not found at {manifest_path}. Cannot use constraint loss for binary encoding."
                )
                sys.exit(1)
            try:
                validator = BlocksWorldValidator(args.num_blocks, args.encoding_type, predicate_manifest_file=manifest_path)
                print("Validator for binary encoding initialized successfully.")
            except Exception as e:
                print(f"ERROR: Failed to initialize validator: {e}")
                sys.exit(1)
        elif args.encoding_type == "sas":
            # SAS validator doesn't need a manifest file.
            print("WARNING: Constraint loss for SAS encoding is not implemented and will have no effect.")
            validator = BlocksWorldValidator(args.num_blocks, args.encoding_type)
            print("Validator for SAS encoding initialized.")

    # Load Datasets (PaTSDataset is generic)
    print("Loading datasets...")
    try:
        train_dataset = PaTSDataset(
            dataset_dir=args.dataset_dir, split_file_name="train_files.txt", encoding_type=args.encoding_type
        )
        val_dataset = PaTSDataset(
            dataset_dir=args.dataset_dir, split_file_name="val_files.txt", encoding_type=args.encoding_type
        )
    except Exception as e:
        print(f"Error initializing PaTSDataset: {e}")
        sys.exit(1)

    if train_dataset.state_dim is None or train_dataset.state_dim <= 0:
        print(
            f"Could not determine num_features for {args.num_blocks} blocks from {args.dataset_dir}. Check dataset integrity and paths."
        )
        sys.exit(1)
    num_features = train_dataset.state_dim
    print(f"Number of features (state_dim) from dataset: {num_features}")

    if len(train_dataset) == 0:
        print(
            f"Training dataset for N={args.num_blocks} is empty. Check {args.dataset_split_dir / 'train_files.txt'}. Exiting."
        )
        sys.exit(1)
    if len(val_dataset) == 0:
        print(
            f"Warning: Validation dataset for N={args.num_blocks} is empty. Check {args.dataset_split_dir / 'val_files.txt'}."
        )

    if args.model_type == "lstm":
        # Use partial to create a collate function with the mlm_mask_prob argument
        collate_fn_with_args = partial(lstm_collate_fn, mlm_mask_prob=args.mlm_mask_prob if args.use_mlm_task else 0.0)

        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_with_args, num_workers=0
        )
        val_dataloader = (
            DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_with_args, num_workers=0
            )
            if len(val_dataset) > 0
            else None
        )

        model = PaTS_LSTM(
            num_features=num_features,
            hidden_size=args.lstm_hidden_size,
            num_lstm_layers=args.lstm_num_layers,
            dropout_prob=args.lstm_dropout_prob,
            use_mlm_task=args.use_mlm_task if args.encoding_type == "binary" else False,  # Disable MLM for SAS+
            encoding_type=args.encoding_type,
            num_blocks=args.num_blocks,
            embedding_dim=args.lstm_embedding_dim,
        ).to(DEVICE)
        setattr(model, "target_num_blocks", args.num_blocks)
        print(model)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable LSTM parameters: {total_params}")

        lstm_model_save_path = model_specific_output_dir / f"pats_lstm_model_N{args.num_blocks}.pth"
        train_lstm_model_loop(model, train_dataloader, val_dataloader, validator, args, num_features, lstm_model_save_path)

    elif args.model_type == "ttm":
        print("Starting TTM training setup...")
        ttm_log_file = model_specific_output_dir / f"ttm_training_N{args.num_blocks}.log"
        ttm_setup_logging(args.log_level, ttm_log_file)

        max_plan_len_in_train_data = 0
        if len(train_dataset.basenames) > 0:
            for basename_for_len_check in train_dataset.basenames:  # Iterate through filtered basenames
                # Construct full path to .npy file using dataset_dir structure
                traj_file_path_for_len = args.dataset_dir / "trajectories_bin" / f"{basename_for_len_check}.traj.bin.npy"
                if traj_file_path_for_len.exists():
                    try:
                        traj_np = torch.from_numpy(np.load(traj_file_path_for_len))
                        if traj_np is not None and traj_np.ndim == 2:
                            max_plan_len_in_train_data = max(max_plan_len_in_train_data, traj_np.shape[0])
                    except Exception as e:
                        print(f"Warning: Could not load trajectory {traj_file_path_for_len} to determine max length: {e}")
                # else: # This can be too verbose if many files are not found (e.g. if basenames are not filtered)
                # print(f"Warning: Trajectory file {traj_file_path_for_len} not found during max length check.")

        print(f"Max plan length in training data for N={args.num_blocks}: {max_plan_len_in_train_data}")

        final_candidate = determine_ttm_model(
            max_plan_length=max_plan_len_in_train_data if max_plan_len_in_train_data > 0 else 60,
            recommended_prediction_length=(max_plan_len_in_train_data + 2) if max_plan_len_in_train_data > 0 else 60,
            user_context_length=args.context_length,
            user_prediction_length=args.prediction_length,
        )

        if final_candidate is None:
            print("ERROR: determine_ttm_model returned None. Please check your input parameters or implementation.")
            sys.exit(1)

        auto_context_length, auto_prediction_length = (
            final_candidate["context_length"],
            final_candidate["prediction_length"],
        )

        ttm_model_config = TTMModelConfig(
            context_length=int(auto_context_length),
            prediction_length=int(auto_prediction_length),
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            ttm_model_path=args.ttm_model_path,
            seed=args.seed,
            state_dim=num_features,
            encoding_type=args.encoding_type,
            num_blocks=args.num_blocks,
        )
        print(f"TTM ModelConfig: {ttm_model_config}")

        # Pass model_specific_output_dir to BlocksWorldTTM for its internal logging/output paths
        ttm_trainer_instance = BlocksWorldTTM(
            model_config=ttm_model_config, device=DEVICE, output_dir=model_specific_output_dir
        )

        # Pass PaTSDataset instances directly
        ttm_trainer_instance.train(train_dataset, val_dataset if len(val_dataset) > 0 else None)

        final_model_assets_path = model_specific_output_dir / "final_model_assets"
        if final_model_assets_path.exists():
            print(f"TTM model save path {final_model_assets_path} already exists. Appending timestamp.")
            final_model_assets_path = final_model_assets_path.with_name(
                f"final_model_assets_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

        print(f"Saving TTM model assets to {final_model_assets_path}")
        ttm_trainer_instance.save(final_model_assets_path)
        print("TTM Training finished.")

    elif args.model_type == "llama_finetune":
        print("Starting Llama fine-tuning setup...")

        # Load base model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.llama_model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # Llama-3-Instruct prefers left padding for generation

        model = AutoModelForCausalLM.from_pretrained(
            args.llama_model_id,
            torch_dtype=torch.bfloat16,  # As requested, no quantization
            device_map="auto",
        )

        # Prepare model for LoRA training (e.g., gradient checkpointing, fp32 for layer norm)
        # Even without kbit, this helps with memory and stability
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

        # LoRA configuration
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",  # Common for LoRA
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules.split(","),  # Split comma-separated string
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Prepare datasets for Llama fine-tuning
        train_llama_dataset = LlamaFinetuneDataset(
            pats_dataset=train_dataset,
            tokenizer=tokenizer,
            encoding_type=args.encoding_type,
            num_blocks=args.num_blocks,
            max_seq_len=args.llama_max_seq_len,
        )
        val_llama_dataset = (
            LlamaFinetuneDataset(
                pats_dataset=val_dataset,
                tokenizer=tokenizer,
                encoding_type=args.encoding_type,
                num_blocks=args.num_blocks,
                max_seq_len=args.llama_max_seq_len,
            )
            if len(val_dataset) > 0
            else None
        )

        data_collator = LlamaFinetuneDataCollator(tokenizer, max_seq_len=args.llama_max_seq_len)

        # Training arguments for HuggingFace Trainer
        training_args = TrainingArguments(
            output_dir=model_specific_output_dir / "lora_checkpoints",  # Checkpoints will be saved here
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            optim="paged_adamw_8bit"
            if torch.cuda.is_available()
            else "adamw_torch",  # Use paged_adamw_8bit if available, else adamw_torch
            learning_rate=args.learning_rate,
            fp16=False,  # We are using bfloat16, not fp16
            bf16=True,  # Enable bfloat16
            max_grad_norm=0.3,  # Common for LLMs
            warmup_ratio=args.warmup_ratio,
            lr_scheduler_type="cosine",
            logging_steps=args.logging_steps,
            save_strategy="steps",
            save_steps=args.save_steps,
            eval_strategy="steps" if val_llama_dataset else "no",
            eval_steps=args.eval_steps if val_llama_dataset else None,
            report_to="tensorboard",
            load_best_model_at_end=True if val_llama_dataset else False,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            seed=args.seed,
            ddp_find_unused_parameters=False,  # Important for LoRA
            remove_unused_columns=False,  # Keep all columns for custom collator
            weight_decay=args.weight_decay,
        )

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_llama_dataset,
            eval_dataset=val_llama_dataset,
            data_collator=data_collator,
        )

        # Start training
        print("Starting Llama LoRA fine-tuning...")
        trainer.train()
        print("Llama LoRA fine-tuning finished.")

        # Save the fine-tuned adapter
        lora_adapter_output_path = model_specific_output_dir / "lora_adapter"
        lora_adapter_output_path.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(lora_adapter_output_path)
        tokenizer.save_pretrained(lora_adapter_output_path)  # Save tokenizer with adapter
        print(f"LoRA adapter saved to {lora_adapter_output_path}")

    elif args.model_type == "xgboost":
        print("Starting XGBoost training setup...")

        # 1. Prepare data in tabular format
        print("Preparing data for XGBoost...")
        X_train, y_train = prepare_data_for_xgboost(train_dataset, args.xgboost_context_window_size)

        if X_train.shape[0] == 0:
            print("ERROR: No training data could be generated for XGBoost. Exiting.")
            sys.exit(1)

        # 2. Initialize and train the model
        planner = XGBoostPlanner(
            encoding_type=args.encoding_type,
            num_blocks=args.num_blocks,
            seed=args.seed,
            context_window_size=args.xgboost_context_window_size,
        )

        planner.train(X_train, y_train)

        # 3. Save the model
        model_save_path = model_specific_output_dir / f"pats_xgboost_model_N{args.num_blocks}.joblib"
        planner.save(model_save_path)
        print("XGBoost training finished.")

    else:
        print(f"Unknown model type: {args.model_type}")
        sys.exit(1)

    print(f"Training for {args.model_type} N={args.num_blocks} complete. Outputs in {model_specific_output_dir}")


if __name__ == "__main__":
    main()
