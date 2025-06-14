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

(Assumes dataset for N=4 blocks has been generated and `data/blocks_4/train_files.txt` etc. exist, and TTM uses a JSON file that lists these trajectories)

The TTM script (`scripts/models/ttm.py`) expects a JSON dataset file that aggregates information from multiple problem instances. You would typically create this JSON file as a preprocessing step, listing paths to the `.traj.bin.npy` and `.goal.bin.npy` files, or embedding the data directly.

Let's assume you have `data/blocks_4/blocksworld_N4_ttm_dataset.json`.

```bash
# From the project root directory
uv run python scripts/models/ttm.py train \
    --dataset_file data/blocks_4/blocksworld_N4_ttm_dataset.json \
    --output_dir ./output_ttm_N4 \
    --num_epochs 100 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --context_length 60 \
    --prediction_length 60 \
    --seed 42
```

This will train a TTM model and save its assets (weights, config) into `./output_ttm_N4/final_model_assets/`.

### 3. Training a Model (Example: LSTM)

(Assumes dataset for N=4 blocks has been generated and `data/blocks_4/train_files.txt` etc. exist)

The LSTM script (`scripts/models/lstm.py`) directly uses the split files.

```bash
# From the project root directory
uv run python scripts/models/lstm.py \
    data/blocks_4 \
    data/blocks_4 \
    ./output_lstm_N4 \
    --num_blocks 4 \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.001
```

- The first `data/blocks_4` is `dataset_dir` (where `trajectories_bin` is).
- The second `data/blocks_4` is `dataset_split_dir` (where `train_files.txt` is).
- `./output_lstm_N4` is the output directory for the model.
  This will train an LSTM model and save it as `./output_lstm_N4/pats_lstm_model_N4.pth`.

_(Note: The LSTM script's argument parsing might need slight adjustments to match this example if it was changed; the example above assumes it can take `num_blocks`, `epochs`, etc., as CLI args, and that the positional args are for dataset paths.)_

### 4. Benchmarking a Trained Model (Example: TTM for N=4)

```bash
# From the project root directory
uv run python scripts/benchmark.py \
    --dataset_dir ./data \
    --num_blocks 4 \
    --model_type ttm \
    --model_path ./output_ttm_N4/final_model_assets \
    --output_dir ./benchmark_results \
    --max_plan_length 60 \
    --save_detailed_results
```

### 5. Benchmarking a Trained Model (Example: LSTM for N=4)

```bash
# From the project root directory
uv run python scripts/benchmark.py \
    --dataset_dir ./data \
    --num_blocks 4 \
    --model_type lstm \
    --model_path ./output_lstm_N4/pats_lstm_model_N4.pth \
    --output_dir ./benchmark_results \
    --max_plan_length 60 \
    --save_detailed_results
```

The benchmark results (aggregated JSON and optionally detailed JSON) will be saved in `./benchmark_results/`.
