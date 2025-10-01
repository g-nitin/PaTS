import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers.trainer_utils import set_seed as ttm_set_seed

from scripts.models.lstm import PaTS_LSTM, lstm_collate_fn
from scripts.models.ttm import BlocksWorldTTM, determine_ttm_model
from scripts.models.ttm import ModelConfig as TTMModelConfig
from scripts.models.ttm import setup_logging as ttm_setup_logging
from scripts.models.xgboost import XGBoostPlanner
from scripts.pats_dataset import PaTSDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_forecast_losses: List[float],
    val_forecast_losses: List[float],
    epochs: int,
    model_type: str,
    encoding_type: str,
    num_blocks: int,
    output_dir: Path,
):
    """
    Plots and saves the training and validation loss curves.
    """
    if not train_losses or not val_losses:
        print("No loss data to plot.")
        return

    sns.set_theme()  # Apply seaborn theme for aesthetics

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, epochs + 1), train_losses, label="Train Total Loss")
    plt.plot(range(1, epochs + 1), val_losses, label="Validation Total Loss")
    plt.plot(range(1, epochs + 1), train_forecast_losses, label="Train Forecast Loss", linestyle="--")
    plt.plot(range(1, epochs + 1), val_forecast_losses, label="Validation Forecast Loss", linestyle="--")

    plt.title(f"{model_type} Training & Validation Loss (N={num_blocks}, Encoding={encoding_type})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plot_filename = f"training_loss_{model_type}_N{num_blocks}_{encoding_type}.png"
    plt.savefig(output_dir / plot_filename)
    plt.close()  # Close the plot to free memory
    print(f"Training loss plot saved to {output_dir / plot_filename}")


def train_lstm_model_loop(model, train_loader, val_loader, args, num_features, model_save_path):
    """
    Train the provided LSTM-based PaTS model over multiple epochs.
    This function runs a full training loop over the given `train_loader`, evaluates on `val_loader` each epoch,
    and saves the best model (by validation loss) to `model_save_path`.
    It supports two encoding modes:
        - "sas" (SAS+): forecasting is performed as a multi-class classification over locations for each block. CrossEntropyLoss is used and MLM / constraint losses are disabled.
        - "bin": forecasting is performed as independent binary predictions per block-location via BCEWithLogitsLoss. Similar to One-hot encoding. Optional MLM and constraint violation losses may be included.
    Uses AdamW optimizer and ReduceLROnPlateau scheduler (monitoring validation loss).

    - For binary encoding:
        - Forecasting loss is computed per-element using BCEWithLogitsLoss with reduction="none" and then averaged over only the valid (un-padded) forecast elements.
        - MLM loss (if args.use_mlm_task) is computed similarly by masking to the mlm_predicate_mask and averaged over masked elements.
        - Forecasting logits and targets are expected to have shape (B, S_max, N_blocks) (logits are raw, targets are 0/1 floats or ints). mlm_logits and mlm_predicate_mask (if used) should align in shape with input_seqs (B, S_max, N_blocks).
    - For SAS+ encoding:
        - Forecasting logits are expected to have shape (B, S_max, N_blocks, N_locs) and targets shape
            (B, S_max, N_blocks) containing integer class labels. Targets are mapped to class indices via model._map_sas_to_indices before applying CrossEntropyLoss.

    - Saves a checkpoint dict containing model/optimizer state_dict, training hyperparameters and metadata whenever a new best validation loss is observed.

    :param model: The PaTS LSTM model to train. Must implement forward(input_seqs, goal_states, lengths) and return a tuple (forecasting_logits, mlm_logits, ...).
        For SAS+ encoding, model must provide model.num_locations and method model._map_sas_to_indices(target_tensor) that maps SAS+ targets to class indices.
    :param train_loader: DataLoader or iterable yielding batches as dictionaries with keys:
        - "input_sequences": tensor, shape (B, S_max, N_blocks) (int for SAS+, float/binary for binary)
        - "goal_states": tensor, shape compatible with model (commonly (B, N_blocks) or (B, S_max, N_blocks))
        - "target_sequences": tensor, shape (B, S_max, N_blocks)
    :param val_loader: Validation DataLoader/iterable with the same batch dictionary format as train_loader. If None, validation is skipped (but scheduler.step will still be called with inf).
    :param args: Configuration with fields (used fields):
        - encoding_type: "sas" or "binary"
        - num_blocks: int (used for SAS+ clamping / metadata)
        - epochs: int
        - learning_rate: float
        - lstm_hidden_size, lstm_num_layers, lstm_dropout_prob, lstm_embedding_dim: training metadata saved in checkpoint
    :param num_features: Number of input features (saved in the checkpoint metadata).
    :param model_save_path: File path where the best model checkpoint dict will be saved via torch.save when validation loss improves.
    :return: None
    """

    print("\nStarting LSTM training...")

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

    # Lists to store losses for plotting
    train_losses_history = []
    val_losses_history = []
    train_forecast_losses_history = []
    val_forecast_losses_history = []

    for epoch in range(args.epochs):
        model.train()
        epoch_train_loss, epoch_forecast_loss = 0.0, 0.0
        num_train_batches = 0

        for batch_data in train_loader:
            if batch_data is None:
                continue
            input_seqs = batch_data["input_sequences"].to(DEVICE)
            goal_states = batch_data["goal_states"].to(DEVICE)
            target_seqs = batch_data["target_sequences"].to(DEVICE)
            lengths = batch_data["lengths"]

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
            forecasting_logits, _ = model(input_seqs, goal_states, lengths)

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
                # Create a mask for valid forecasting steps
                # (B, S_max, N_blocks, N_locs)
                forecasting_mask = torch.zeros_like(target_seqs, dtype=torch.bool).to(DEVICE)
                for i, length_val in enumerate(lengths):
                    if length_val > 0:
                        forecasting_mask[i, :length_val, :] = True  # Mark valid steps

                # Count the number of valid forecasting elements
                num_forecast_elements = forecasting_mask.float().sum()
                if num_forecast_elements == 0:
                    continue

                # Compute the loss
                loss_forecasting_unreduced = criterion(forecasting_logits, target_seqs)
                loss_forecasting = (loss_forecasting_unreduced * forecasting_mask.float()).sum() / num_forecast_elements

            # Total Loss
            total_loss = loss_forecasting
            total_loss.backward()
            if args.clip_grad_norm is not None and args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

            epoch_train_loss += total_loss.item()
            epoch_forecast_loss += loss_forecasting.item()
            num_train_batches += 1

        avg_train_loss = epoch_train_loss / num_train_batches if num_train_batches > 0 else float("inf")
        avg_forecast_loss = epoch_forecast_loss / num_train_batches if num_train_batches > 0 else float("inf")

        # Validation
        model.eval()
        epoch_val_loss, epoch_val_forecast_loss = 0.0, 0.0
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

                    if args.encoding_type == "sas":
                        input_seqs.clamp_(min=0, max=args.num_blocks)
                        goal_states.clamp_(min=0, max=args.num_blocks)

                    forecasting_logits, _ = model(input_seqs, goal_states, lengths)

                    # Create a mask for valid forecasting steps
                    # (B, S_max, N_blocks, N_locs)
                    if args.encoding_type == "sas":
                        mask = (
                            torch.arange(max(lengths), device=DEVICE)[None, :]
                            < lengths.clone().detach().to(DEVICE)[:, None]
                        )
                        active_logits = forecasting_logits[mask]
                        active_targets = target_seqs[mask]

                        if active_targets.numel() == 0:
                            continue

                        # Map active targets to their indices
                        active_targets_indices = model._map_sas_to_indices(active_targets)
                        loss_forecasting = criterion(
                            active_logits.reshape(-1, model.num_locations), active_targets_indices.reshape(-1)
                        )

                    else:  # Binary
                        # (B, S_max, N_blocks, N_locs)
                        forecasting_mask = torch.zeros_like(target_seqs, dtype=torch.bool).to(DEVICE)
                        for i, length_val in enumerate(lengths):
                            if length_val > 0:
                                forecasting_mask[i, :length_val, :] = True

                        # Count the number of valid forecasting elements
                        num_forecast_elements = forecasting_mask.float().sum()
                        if num_forecast_elements == 0:
                            continue

                        # Compute the loss for the valid forecasting elements
                        loss_forecasting_unreduced = criterion(forecasting_logits, target_seqs)
                        loss_forecasting = (
                            loss_forecasting_unreduced * forecasting_mask.float()
                        ).sum() / num_forecast_elements

                    total_loss = loss_forecasting
                    epoch_val_loss += total_loss.item()
                    epoch_val_forecast_loss += loss_forecasting.item()
                    num_val_batches += 1

        avg_val_loss = epoch_val_loss / num_val_batches if num_val_batches > 0 else float("inf")
        avg_val_forecast_loss = epoch_val_forecast_loss / num_val_batches if num_val_batches > 0 else float("inf")
        scheduler.step(avg_val_loss)

        # Store losses for plotting
        train_losses_history.append(avg_train_loss)
        val_losses_history.append(avg_val_loss)
        train_forecast_losses_history.append(avg_forecast_loss)
        val_forecast_losses_history.append(avg_val_forecast_loss)

        train_loss_str = f"Train Loss: {avg_train_loss:.4f} (F: {avg_forecast_loss:.4f})"
        val_loss_str = f"Val Loss: {avg_val_loss:.4f} (F: {avg_val_forecast_loss:.4f})"
        print(f"Epoch [{epoch + 1}/{args.epochs}] {train_loss_str}, {val_loss_str}")

        # Save best model
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
                    "target_num_blocks": args.num_blocks,
                    "embedding_dim": args.lstm_embedding_dim if args.encoding_type == "sas" else None,
                },
                model_save_path,
            )
            print(f"Model saved to {model_save_path} (Val Loss: {best_val_loss:.4f})")

    print("LSTM Training finished.")

    # Call plotting function after training
    plot_training_curves(
        train_losses_history,
        val_losses_history,
        train_forecast_losses_history,
        val_forecast_losses_history,
        args.epochs,
        args.model_type,
        args.encoding_type,
        args.num_blocks,
        model_save_path.parent,  # Pass the directory where the model is saved
    )


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
            # current_state = trajectory[t]
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
    print("\nStarting unified training script for PaTS models...")
    parser = argparse.ArgumentParser(description="Unified Training Script for PaTS Models")

    # Common arguments (agnostic to `model_type`)
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["lstm", "ttm", "xgboost"],
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
    parser.add_argument(
        "--num_blocks",
        type=int,
        required=True,
        help="Number of blocks for this training run.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Base directory to save trained models and logs.",
    )
    parser.add_argument(
        "--encoding_type",
        type=str,
        default="bin",
        choices=["bin", "sas"],
        help="The encoding type of the dataset to use.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs. Default is 100.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size. Default is 32.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate. Default is 1e-3.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for reproducibility. Default is 13.",
    )

    # LSTM specific arguments
    parser.add_argument(
        "--lstm_hidden_size",
        type=int,
        default=128,
        help="Hidden size for LSTM. Default is 128.",
    )
    parser.add_argument(
        "--lstm_num_layers",
        type=int,
        default=2,
        help="Number of LSTM layers. Default is 2.",
    )
    parser.add_argument(
        "--lstm_dropout_prob",
        type=float,
        default=0.2,
        help="Dropout probability for LSTM. Default is 0.2.",
    )
    parser.add_argument(
        "--lstm_embedding_dim",
        type=int,
        default=32,
        help="Embedding dimension for SAS+ encoding. Default is 32.",
    )
    parser.add_argument(
        "--clip_grad_norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm for LSTM (0 to disable). Default is 1.0",
    )

    # TTM specific arguments
    parser.add_argument(
        "--ttm_model_path",
        type=str,
        default="ibm-granite/granite-timeseries-ttm-r2.1",
        help="Base TTM model path from HuggingFace or local. Default is 'ibm-granite/granite-timeseries-ttm-r2.1'.",
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

    args = parser.parse_args()

    print("Parsed Arguments:")
    pprint(vars(args))
    print()

    # lstm_param_names = [
    #     args.lstm_hidden_size,
    #     args.lstm_num_layers,
    #     args.lstm_dropout_prob,
    #     args.lstm_embedding_dim,
    #     args.clip_grad_norm,
    # ]
    # ttm_param_names = [
    #     args.ttm_model_path,
    #     args.context_length,
    #     args.prediction_length,
    #     args.log_level,
    # ]
    # xgboost_param_names = [args.xgboost_context_window_size]
    # if (  # Confirm if `model_type`=='lstm' that non-lstm related parameters are not given
    #     args.model_type == "lstm"
    #     and any(param is not None for param in ttm_param_names)
    #     and any(param is not None for param in xgboost_param_names)
    # ):
    #     sys.exit("TTM and XGBoost related arguments are not applicable for LSTM. Exiting.")

    # if (  # Confirm if `model_type`=='xgboost' that non-xgboost related parameters are not given
    #     args.model_type == "xgboost"
    #     and any(param is not None for param in lstm_param_names)
    #     and any(param is not None for param in ttm_param_names)
    # ):
    #     sys.exit("LSTM and TTM related arguments are not applicable for XGBoost. Exiting.")

    # if (  # Confirm if `model_type`=='ttm' that non-ttm related parameters are not given
    #     args.model_type == "ttm"
    #     and any(param is not None for param in lstm_param_names)
    #     and any(param is not None for param in xgboost_param_names)
    # ):
    #     sys.exit("LSTM and XGBoost related arguments are not applicable for TTM. Exiting.")

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

    # Load Datasets
    # `dataset_dir` is the RAW_BLOCK_DIR (e.g., data/raw_problems/blocksworld/N4)
    # We need to construct the PROCESSED_BLOCK_ENCODING_DIR
    processed_data_for_encoding_dir = (
        args.dataset_dir.parent.parent.parent
        / "processed_trajectories"
        / args.dataset_dir.parent.name
        / args.dataset_dir.name
        / args.encoding_type
    )
    print(f"Loading datasets from {processed_data_for_encoding_dir} based on splits in {args.dataset_split_dir}.")

    try:
        train_dataset = PaTSDataset(
            raw_data_dir=args.dataset_dir,
            processed_data_dir=processed_data_for_encoding_dir,
            split_file_name="train_files.txt",
            encoding_type=args.encoding_type,
        )
        val_dataset = PaTSDataset(
            raw_data_dir=args.dataset_dir,
            processed_data_dir=processed_data_for_encoding_dir,
            split_file_name="val_files.txt",
            encoding_type=args.encoding_type,
        )
    except Exception as e:
        print(f"Error initializing PaTSDataset: {e}")
        sys.exit(1)

    if train_dataset.state_dim is None or train_dataset.state_dim <= 0:
        print(f"Could not determine num_features for {args.num_blocks} blocks from {args.dataset_dir}.")
        print("Check dataset integrity and paths.")
        sys.exit(1)
    num_features = train_dataset.state_dim
    print(f"Number of features (state_dim) from dataset: {num_features}")

    if len(train_dataset) == 0:
        sys.exit(
            f"Training dataset for N={args.num_blocks} is empty. Check {args.dataset_split_dir / 'train_files.txt'}. Exiting."
        )
    if len(val_dataset) == 0:
        sys.exit(
            f"Warning: Validation dataset for N={args.num_blocks} is empty. Check {args.dataset_split_dir / 'val_files.txt'}."
        )

    if args.model_type == "lstm":
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lstm_collate_fn,
            num_workers=0,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lstm_collate_fn,
            num_workers=0,
        )

        model = PaTS_LSTM(
            num_features=num_features,
            hidden_size=args.lstm_hidden_size,
            num_lstm_layers=args.lstm_num_layers,
            dropout_prob=args.lstm_dropout_prob,
            encoding_type=args.encoding_type,
            num_blocks=args.num_blocks,
            embedding_dim=args.lstm_embedding_dim,
        ).to(DEVICE)

        setattr(model, "target_num_blocks", args.num_blocks)  # For saving/loading purposes
        print(model)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable LSTM parameters: {total_params}")

        lstm_model_save_path = model_specific_output_dir / f"pats_lstm_model_N{args.num_blocks}.pth"
        train_lstm_model_loop(
            model,
            train_dataloader,
            val_dataloader,
            args,
            num_features,
            lstm_model_save_path,
        )

    elif args.model_type == "ttm":
        print("Starting TTM training setup...")
        ttm_log_file = model_specific_output_dir / f"ttm_training_N{args.num_blocks}.log"
        ttm_setup_logging(args.log_level, ttm_log_file)

        max_plan_len_in_train_data = 0
        if len(train_dataset.basenames) > 0:
            for basename_for_len_check in train_dataset.basenames:  # Iterate through filtered basenames
                # Construct full path to .npy file using processed_data_for_encoding_dir
                traj_file_path_for_len = (
                    processed_data_for_encoding_dir / f"{basename_for_len_check}.traj.{args.encoding_type}.npy"
                )
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
