import sys
from pathlib import Path

# 1. <.parent> contains this train_model.py
# 2. <.parent.parent> contains scripts/
# 3. <.parent.parent.parent> contains PaTS project
# 4. <.parent.parent.parent.parent> for good measure
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers.trainer_utils import set_seed as ttm_set_seed

from scripts.models.lstm import PaTS_LSTM, lstm_collate_fn
from scripts.models.ttm import BlocksWorldTTM, determine_ttm_lengths
from scripts.models.ttm import ModelConfig as TTMModelConfig
from scripts.models.ttm import setup_logging as ttm_setup_logging
from scripts.pats_dataset import PaTSDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(DEVICE)
exit()


def train_lstm_model_loop(model, train_loader, val_loader, args, num_features, model_save_path):
    print("Starting LSTM training...")
    # Ensure criterion is defined here or passed if it's specific
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    # Increased patience, added verbose
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=10, factor=0.5)

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        epoch_train_loss = 0.0
        num_train_batches = 0

        for batch_idx, batch_data in enumerate(train_loader):
            if batch_data is None:
                continue
            input_seqs = batch_data["input_sequences"].to(DEVICE)
            goal_states = batch_data["goal_states"].to(DEVICE)
            target_seqs = batch_data["target_sequences"].to(DEVICE)
            lengths = batch_data["lengths"]  # Keep on CPU for pack_padded_sequence

            optimizer.zero_grad()
            output_logits, _ = model(input_seqs, goal_states, lengths)  # Pass lengths

            mask = torch.zeros_like(target_seqs, dtype=torch.bool).to(DEVICE)
            for i, length_val in enumerate(lengths):  # Renamed length to length_val to avoid conflict
                if length_val > 0:
                    mask[i, :length_val, :] = True

            if mask.float().sum() == 0:
                # print(f"Warning: Empty mask in training batch {batch_idx}. Skipping.")
                continue

            loss_unreduced = criterion(output_logits, target_seqs)
            loss = (loss_unreduced * mask.float()).sum() / mask.float().sum()

            loss.backward()
            if args.clip_grad_norm is not None and args.clip_grad_norm > 0:  # Check if > 0
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

            epoch_train_loss += loss.item()
            num_train_batches += 1

        avg_epoch_train_loss = epoch_train_loss / num_train_batches if num_train_batches > 0 else float("inf")

        # Validation
        model.eval()
        epoch_val_loss = 0.0
        num_val_batches = 0
        with torch.no_grad():
            for batch_data in val_loader:
                if batch_data is None:
                    continue
                input_seqs = batch_data["input_sequences"].to(DEVICE)
                goal_states = batch_data["goal_states"].to(DEVICE)
                target_seqs = batch_data["target_sequences"].to(DEVICE)
                lengths = batch_data["lengths"]  # Keep on CPU

                output_logits, _ = model(input_seqs, goal_states, lengths)  # Pass lengths
                mask = torch.zeros_like(target_seqs, dtype=torch.bool).to(DEVICE)
                for i, length_val in enumerate(lengths):  # Renamed length to length_val
                    if length_val > 0:
                        mask[i, :length_val, :] = True

                if mask.float().sum() == 0:
                    # print(f"Warning: Empty mask in validation batch. Skipping.") # Optional
                    continue
                loss_unreduced = criterion(output_logits, target_seqs)
                loss = (loss_unreduced * mask.float()).sum() / mask.float().sum()
                epoch_val_loss += loss.item()
                num_val_batches += 1

        avg_epoch_val_loss = epoch_val_loss / num_val_batches if num_val_batches > 0 else float("inf")
        scheduler.step(avg_epoch_val_loss)  # Step scheduler with validation loss

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] Train Loss: {avg_epoch_train_loss:.4f}, Val Loss: {avg_epoch_val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        if avg_epoch_val_loss < best_val_loss:
            best_val_loss = avg_epoch_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_val_loss,
                    "num_features": num_features,
                    "hidden_size": args.lstm_hidden_size,
                    "num_lstm_layers": args.lstm_num_layers,
                    "dropout_prob": args.lstm_dropout_prob,
                    "target_num_blocks": args.num_blocks,
                },
                model_save_path,
            )
            print(f"Model saved to {model_save_path} (Val Loss: {best_val_loss:.4f})")
    print("LSTM Training finished.")


def main():
    parser = argparse.ArgumentParser(description="Unified Training Script for PaTS Models")

    # Common arguments
    parser.add_argument("--model_type", type=str, required=True, choices=["lstm", "ttm"], help="Type of model to train.")
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

    args = parser.parse_args()

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
    print("Loading datasets...")
    try:
        train_dataset = PaTSDataset(dataset_dir=args.dataset_dir, split_file_name="train_files.txt")
        val_dataset = PaTSDataset(dataset_dir=args.dataset_dir, split_file_name="val_files.txt")
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
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lstm_collate_fn, num_workers=0
        )
        # Handle empty val_dataset for val_dataloader
        val_dataloader = (
            DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lstm_collate_fn, num_workers=0)
            if len(val_dataset) > 0
            else None
        )

        model = PaTS_LSTM(
            num_features=num_features,
            hidden_size=args.lstm_hidden_size,
            num_lstm_layers=args.lstm_num_layers,
            dropout_prob=args.lstm_dropout_prob,
        ).to(DEVICE)
        setattr(model, "target_num_blocks", args.num_blocks)  # Store for reference in saved checkpoint
        print(model)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable LSTM parameters: {total_params}")

        lstm_model_save_path = model_specific_output_dir / f"pats_lstm_model_N{args.num_blocks}.pth"
        train_lstm_model_loop(model, train_dataloader, val_dataloader, args, num_features, lstm_model_save_path)

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
                        traj_np = (
                            torch.load(traj_file_path_for_len)
                            if traj_file_path_for_len.suffix == ".pt"
                            else torch.from_numpy(torch.load(traj_file_path_for_len))
                            if traj_file_path_for_len.suffix == ".npy"
                            else None
                        )  # Adjusted for potential .npy loading
                        if traj_np is not None and traj_np.ndim == 2:
                            max_plan_len_in_train_data = max(max_plan_len_in_train_data, traj_np.shape[0])
                    except Exception as e:
                        print(f"Warning: Could not load trajectory {traj_file_path_for_len} to determine max length: {e}")
                # else: # This can be too verbose if many files are not found (e.g. if basenames are not filtered)
                # print(f"Warning: Trajectory file {traj_file_path_for_len} not found during max length check.")

        print(f"Max plan length in training data for N={args.num_blocks}: {max_plan_len_in_train_data}")

        auto_context_length, auto_prediction_length = determine_ttm_lengths(
            max_plan_length=max_plan_len_in_train_data if max_plan_len_in_train_data > 0 else 60,
            recommended_prediction_length=(max_plan_len_in_train_data + 2) if max_plan_len_in_train_data > 0 else 60,
            user_context_length=args.context_length,
            user_prediction_length=args.prediction_length,
        )

        ttm_model_config = TTMModelConfig(
            context_length=auto_context_length,
            prediction_length=auto_prediction_length,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            ttm_model_path=args.ttm_model_path,
            seed=args.seed,
            state_dim=num_features,
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

    else:
        print(f"Unknown model type: {args.model_type}")
        sys.exit(1)

    print(f"Training for {args.model_type} N={args.num_blocks} complete. Outputs in {model_specific_output_dir}")


if __name__ == "__main__":
    main()
