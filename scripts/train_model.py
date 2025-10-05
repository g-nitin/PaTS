import argparse
import sys
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers.trainer_utils import set_seed as ttm_set_seed

from scripts.models.lstm import PaTS_LSTM, lstm_collate_fn, train_lstm_model_loop
from scripts.models.ttm import BlocksWorldTTM, determine_ttm_model
from scripts.models.ttm import ModelConfig as TTMModelConfig
from scripts.models.xgboost import XGBoostPlanner
from scripts.pats_dataset import PaTSDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))


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
        help="Path to the PaTS dataset directory for a specific N (e.g., data/raw_problems/blocksworld/N4).",
    )
    parser.add_argument(
        "--dataset_split_dir",
        type=Path,
        required=True,
        help="Path to the directory containing train_files.txt, etc. (e.g., data/raw_problems/blocksworld/N4/splits).",
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
        default="ibm-granite/granite-timeseries-ttm-r2",
        help="Base TTM model path from HuggingFace or local. Default is `ibm-granite/granite-timeseries-ttm-r2`",
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

    try:
        train_dataset = PaTSDataset(
            raw_data_dir=args.dataset_dir,
            processed_data_dir=args.processed_block_encoding_dir,
            split_file_name="train_files.txt",
            encoding_type=args.encoding_type,
        )
        val_dataset = PaTSDataset(
            raw_data_dir=args.dataset_dir,
            processed_data_dir=args.processed_block_encoding_dir,
            split_file_name="val_files.txt",
            encoding_type=args.encoding_type,
        )
    except Exception as e:
        print(f"Error initializing PaTSDataset: {e}")
        sys.exit(1)

    # Limit dataset for debugging
    # train_dataset.basenames = train_dataset.basenames[:10]
    # val_dataset.basenames = val_dataset.basenames[:5]
    # print(f"~~~~~LIMITING DATASET FOR DEBUGGING~~~~~")

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
            DEVICE,
        )

    elif args.model_type == "xgboost":
        print("\nStarting XGBoost training setup...")

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

    elif args.model_type == "ttm":
        print("Starting TTM training setup...")

        max_plan_len_in_train_data = 0
        if len(train_dataset.basenames) > 0:
            for basename_for_len_check in train_dataset.basenames:  # Iterate through filtered basenames
                # Construct full path to .npy file using processed_data_for_encoding_dir
                traj_file_path_for_len = (
                    args.processed_block_encoding_dir / f"{basename_for_len_check}.traj.{args.encoding_type}.npy"
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
