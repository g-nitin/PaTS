# Planning as Time-Series (PaTS)

## Overview

PaTS explores the idea of treating automated planning as a sequence prediction task, akin to time-series forecasting. Instead of traditional planners that search through a state-space graph using action definitions, PaTS learns from expert-demonstrated trajectories. Given an initial state and a goal state, the trained model "forecasts" a sequence of intermediate states that ideally forms a valid plan to achieve the goal.

This project provides the infrastructure to:

1.  Generate planning problem datasets (currently for Blocks World).
2.  Encode states and trajectories into a binary format suitable for time-series models.
3.  Train time-series models (e.g., TTM, LSTM) to predict state sequences.
4.  Rigorously evaluate the generated plans for validity and goal achievement using a configurable validator and a dedicated benchmarking script.

## [Dataset](data/README.md)

## [Workflow](scripts/README.md)

## Running the System

### 1. Generating the Dataset

(Assumes `generate_dataset.sh` is configured correctly with paths to FD, VAL, etc.)

```bash
# Navigate to the data directory
cd data

# Execute the script (it will create subdirectories like blocks_4/)
# You might be prompted to overwrite if directories exist.
./generate_dataset.sh
# This will generate data for the N_BLOCKS range specified in the script.
# For example, if MIN_BLOCKS_TO_GENERATE=4 and MAX_BLOCKS_TO_GENERATE=4,
# it will create data in ./data/blocks_4/
```

After generation, `data/blocks_<N>/` will contain the PDDL files, plans, VAL logs, encoded trajectories (`.traj.bin.npy`), goals (`.goal.bin.npy`), the `predicate_manifest_<N>.txt`, and split files (`train_files.txt`, etc.).

### 2. Training a Model (Example: TTM)

(Assumes dataset for N=4 blocks has been generated and `data/blocks_4/train_files.txt`, etc. exist)

```bash
# From the project root directory
uv run python -m scripts.train_model \
    --model_type ttm \
    --dataset_dir data/blocks_4 \
    --dataset_split_dir data/blocks_4 \
    --num_blocks 4 \
    --encoding_type bin \ # Specify the encoding: 'bin' or 'sas'
    --output_dir ./training_outputs \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --seed 13
    # --ttm_model_path "ibm-granite/granite-timeseries-ttm-r2" # Optional: specify base TTM model
    # --context_length 60 \ # Optional: TTM can auto-determine this if not provided
    # --prediction_length 60 \ # Optional: TTM can auto-determine this if not provided
```

This will train a TTM model and save its assets (weights, config, logs) into `./training_outputs/ttm_N4/`.
The final model specifically will be in a subdirectory like `final_model_assets/`.

### 3. Training a Model (Example: LSTM)

(Assumes dataset for N=4 blocks has been generated and `data/blocks_4/train_files.txt` etc. exist)

```bash
# From the project root directory
uv run python -m scripts.train_model \
    --model_type lstm \
    --dataset_dir data/blocks_4 \
    --dataset_split_dir data/blocks_4 \
    --num_blocks 4 \
    --encoding_type bin \ # Specify the encoding: 'bin' or 'sas'
    --output_dir ./training_outputs \
    --epochs 200 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --seed 13 \
    --use_constraint_loss \  # Optional: Enable the auxiliary constraint violation loss to enforce domain rules.
    --constraint_loss_weight 1.0 \ # Optional: Controls the influence of the constraint loss.
    --use_mlm_task \  #  Optional: Enables the auxiliary masking task
    --mlm_loss_weight 0.2 \  # Optional: Controls the influence of the MLM loss.
    --mlm_mask_prob 0.15  # Optional: The probability that any given predicate in the input sequence will be chosen for the MLM loss calculation.
    # --lstm_hidden_size 128 \ # Optional
    # --lstm_num_layers 2 \ # Optional
    # --clip_grad_norm 1.0 \ # Optional
```

This will train an LSTM model and save its checkpoint (`.pth`) into `./training_outputs/lstm_N4/`.

### 4. Benchmarking a Trained Model (Example: TTM for N=4)

```bash
# From the project root directory
uv run python -m scripts.benchmark \
    --dataset_dir data/blocks_4 \
    --num_blocks 4 \
    --model_type ttm \
    --model_path ./training_outputs/ttm_N4/final_model_assets \
    --encoding_type bin \ # Must match the encoding the model was trained on
    --output_dir ./benchmark_results \
    --max_plan_length 60 \
    --save_detailed_results
```

### 5. Benchmarking a Trained Model (Example: LSTM for N=4)

```bash
# From the project root directory
uv run python -m scripts.benchmark \
    --dataset_dir data/blocks_4 \
    --num_blocks 4 \
    --model_type lstm \
    --model_path ./training_outputs/lstm_N4/pats_lstm_model_N4.pth \
    --encoding_type bin \ # Must match the encoding the model was trained on
    --output_dir ./benchmark_results \
    --max_plan_length 60 \
    --save_detailed_results
```

The benchmark results (aggregated JSON and optionally detailed JSON) will be saved in `./benchmark_results/`.
