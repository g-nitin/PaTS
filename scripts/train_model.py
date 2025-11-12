import argparse
import sys
import warnings

# from datetime import datetime
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers.trainer_utils import set_seed as ttm_set_seed

from scripts.models.lstm import PaTS_LSTM, lstm_collate_fn, train_lstm_model_loop
from scripts.models.ttm import ModelConfig as TTMModelConfig
from scripts.models.ttm import PaTS_TTM, determine_ttm_model
from scripts.models.xgboost import XGBoostPlanner, prepare_data_for_xgboost
from scripts.pats_dataset import PaTSDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))


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
        help="Path to the raw problem data directory for a specific config (e.g., 'data/raw_problems/blocksworld/N4').",
    )
    parser.add_argument(
        "--processed_encoding_dir",
        type=Path,
        required=True,
        help="Path to the processed encoding directory (e.g., 'data/processed_trajectories/blocksworld/N4/bin')",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="blocksworld",
        choices=["blocksworld", "grippers"],
        help="The planning domain.",
    )
    parser.add_argument(
        "--num_blocks", type=int, help="Number of blocks for the BlocksWorld domain. Required if --domain is 'blocksworld'."
    )
    parser.add_argument(
        "--num-robots", type=int, help="Number of robots for Grippers domain. Required if --domain is 'grippers'."
    )
    parser.add_argument(
        "--num-objects", type=int, help="Number of objects for Grippers domain. Required if --domain is 'grippers'."
    )
    parser.add_argument(
        "--num-rooms", type=int, help="Number of rooms for Grippers domain. Required if --domain is 'grippers'."
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

    if args.domain == "blocksworld" and args.num_blocks is None:
        parser.error("--num_blocks is required when --domain is 'blocksworld'.")
    elif args.domain == "grippers" and (args.num_robots is None or args.num_objects is None or args.num_rooms is None):
        parser.error("--num-robots, --num-objects, and --num-rooms are required when --domain is 'grippers'.")

    print("Parsed Arguments:")
    pprint(vars(args))
    print()

    print(f"Using device: {DEVICE}")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.model_type == "ttm":
        ttm_set_seed(args.seed)  # Specific seed setting for TTM/HuggingFace Trainer

    # Determine a config name for the output directory
    if args.domain == "blocksworld":
        config_name = f"N{args.num_blocks}"
    elif args.domain == "grippers":
        config_name = f"R{args.num_robots}-O{args.num_objects}-RM{args.num_rooms}"
    else:
        warnings.warn(f"Unknown domain {args.domain}. Exiting.")
        sys.exit(1)

    # Create model-specific output directory: <output_dir>/<model_type>_<config_name>/
    model_specific_output_dir = args.output_dir / f"{args.model_type}_{config_name}"
    model_specific_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model outputs will be saved to: {model_specific_output_dir}")

    try:
        train_dataset = PaTSDataset(
            raw_data_dir=args.dataset_dir,
            processed_data_dir=args.processed_encoding_dir,
            split_file_name="train_files.txt",
            encoding_type=args.encoding_type,
        )
        val_dataset = PaTSDataset(
            raw_data_dir=args.dataset_dir,
            processed_data_dir=args.processed_encoding_dir,
            split_file_name="val_files.txt",
            encoding_type=args.encoding_type,
        )
    except Exception as e:
        print(f"Error initializing PaTSDataset: {e}")
        sys.exit(1)

    # Limit dataset for debugging
    # train_dataset.basenames = train_dataset.basenames[:3]
    # val_dataset.basenames = val_dataset.basenames[:2]
    # print("~~~~~LIMITING DATASET FOR DEBUGGING~~~~~")

    if train_dataset.state_dim is None or train_dataset.state_dim <= 0:
        print(f"Could not determine num_features for {args.num_blocks} blocks from {args.dataset_dir}.")
        print("Check dataset integrity and paths.")
        sys.exit(1)
    num_features = train_dataset.state_dim
    print(f"Number of features (state_dim) from dataset: {num_features}")

    if len(train_dataset) == 0:
        sys.exit(
            f"Training dataset for N={args.num_blocks} is empty. Check {args.dataset_dir / 'train_files.txt'}. Exiting."
        )
    if len(val_dataset) == 0:
        sys.exit(
            f"Warning: Validation dataset for N={args.num_blocks} is empty. Check {args.dataset_dir / 'val_files.txt'}."
        )

    # Create the domain_config:
    domain_config = {}
    if args.domain == "blocksworld":
        domain_config["num_blocks"] = args.num_blocks
    elif args.domain == "grippers":
        if not all([args.num_robots, args.num_objects, args.num_rooms]):
            sys.exit("Error: --num-robots, --num-objects, and --num-rooms are required for the grippers domain.")
        domain_config["robots"] = args.num_robots
        domain_config["objects"] = args.num_objects
        domain_config["rooms"] = args.num_rooms

    # Starting training based on LSTM
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
            domain=args.domain,
            domain_config=domain_config,
            embedding_dim=args.lstm_embedding_dim,
        ).to(DEVICE)

        print(model)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable LSTM parameters: {total_params}")

        lstm_model_save_path = model_specific_output_dir / f"pats_lstm_model_{config_name}.pth"
        train_lstm_model_loop(
            model,
            train_dataloader,
            val_dataloader,
            args,
            domain_config,
            num_features,
            config_name,
            lstm_model_save_path,
            DEVICE,
        )

    elif args.model_type == "xgboost":
        print("\nStarting XGBoost training setup...")
        X_train, y_train = prepare_data_for_xgboost(train_dataset, args.xgboost_context_window_size)
        if X_train.shape[0] == 0:
            sys.exit("ERROR: No training data could be generated for XGBoost. Exiting.")

        planner = XGBoostPlanner(
            encoding_type=args.encoding_type,
            domain_config=domain_config,
            seed=args.seed,
            context_window_size=args.xgboost_context_window_size,
        )
        planner.train(X_train, y_train)
        model_save_path = model_specific_output_dir / f"pats_xgboost_model_{config_name}.joblib"
        planner.save(model_save_path)
        print("XGBoost training finished.")

    elif args.model_type == "ttm":
        print("Starting TTM training setup...")
        max_plan_len_in_train_data = max(
            np.load(args.processed_encoding_dir / f"{b}.traj.{args.encoding_type}.npy").shape[0]
            for b in train_dataset.basenames
        )
        print(f"Max plan length in training data for {config_name}: {max_plan_len_in_train_data}")

        final_candidate = determine_ttm_model(
            max_plan_length=max_plan_len_in_train_data,
            recommended_prediction_length=max_plan_len_in_train_data + 2,
            user_context_length=args.context_length,
            user_prediction_length=args.prediction_length,
        )
        if final_candidate is None:
            sys.exit("ERROR: Could not determine a suitable TTM model.")

        ttm_model_config = TTMModelConfig(
            context_length=int(final_candidate["context_length"]),
            prediction_length=int(final_candidate["prediction_length"]),
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            ttm_model_path=args.ttm_model_path,
            seed=args.seed,
            state_dim=num_features,
            encoding_type=args.encoding_type,
            domain_config=domain_config,
        )
        print(f"TTM ModelConfig: {ttm_model_config}")

        ttm_trainer_instance = PaTS_TTM(model_config=ttm_model_config, device=DEVICE, output_dir=model_specific_output_dir)
        ttm_trainer_instance.train(train_dataset, val_dataset if len(val_dataset) > 0 else None)
        final_model_assets_path = model_specific_output_dir / "final_model_assets"
        ttm_trainer_instance.save(final_model_assets_path)
        print("TTM Training finished.")

    else:
        print(f"Unknown model type: {args.model_type}")
        sys.exit(1)

    print(f"Training for {args.model_type} ({config_name}) complete. Outputs in {model_specific_output_dir}")


if __name__ == "__main__":
    main()
