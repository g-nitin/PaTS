# Planning as Time-Series (PaTS)

## Overview

This project introduces **Planning as Time-Series (PaTS)**, a novel framework that reformulates automated planning as a time-series forecasting problem. Instead of traditional search-based methods, PaTS learns an implicit transition model from expert-demonstrated trajectories. Given an initial state and a goal, a trained time-series model "forecasts" a sequence of intermediate states that forms a valid plan.

This repository provides the complete infrastructure to:

1.  **Generate Datasets**: Create planning problem datasets for the Blocksworld domain with varying complexity.
2.  **Encode States**: Convert symbolic planning states into numerical vectors using either a sparse binary predicate representation (`bin`) or a compact position-based one (`sas`).
3.  **Train Models**: Train a diverse range of models, including LSTMs, modern MLP-based architectures like TTM, and classical gradient-boosted models like XGBoost.
4.  **Benchmark Performance**: Rigorously evaluate the generated plans for validity, correctness, and optimality using a dedicated benchmarking script.

## [Dataset Generation & Structure](data/README.md)

Details on how to generate data and the structure of the output can be found in the data directory's README.

## [Workflow & Scripts](scripts/README.md)

A detailed explanation of the workflow, key scripts, and evaluation metrics is available in the scripts directory's README.

## Running the System

### 1. Generating the Dataset

First, configure the paths in `data/generate_dataset.sh`. You can also set the desired encoding type (`bin` or `sas`). The script will create a directory like `data/blocks_4-sas/`.

```bash
# Navigate to the data directory
cd data

# Execute the script for SAS+ encoding
# The script can be modified to change encoding type or other parameters.
./generate_dataset.sh
```

### 2. Training a Model

The `scripts/train_model.py` script is the unified entry point for training all applicable models.

**Example: Training an LSTM with SAS+ encoding**

```bash
# From the project root directory
uv run python -m scripts.train_model \
    --model_type lstm \
    --dataset_dir data/blocks_4-sas \
    --num_blocks 4 \
    --encoding_type sas \
    --output_dir ./training_outputs \
    --epochs 400 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --seed 13
```

**Example: Training an XGBoost model with SAS+ encoding and a context window**

```bash
# From the project root directory
uv run python -m scripts.train_model \
    --model_type xgboost \
    --dataset_dir data/blocks_4-sas \
    --num_blocks 4 \
    --encoding_type sas \
    --output_dir ./training_outputs \
    --seed 13 \
    --xgboost_context_window_size 3
```

### 3. Benchmarking a Trained Model

Use `scripts/benchmark.py` to evaluate a trained model on the test set.

**Example: Benchmarking the trained LSTM**

```bash
# From the project root directory
uv run python -m scripts.benchmark \
    --dataset_dir data/blocks_4-sas \
    --num_blocks 4 \
    --model_type lstm \
    --model_path ./training_outputs/lstm_N4/pats_lstm_model_N4.pth \
    --encoding_type sas \
    --output_dir ./benchmark_results \
    --max_plan_length 60 \
    --save_detailed_results
```

### 4. Benchmarking an LLM (Inference Only)

Large Language Models like Llama are used for inference without a separate training step. The `benchmark.py` script can directly evaluate them in zero-shot or few-shot settings.

**Example: Benchmarking Llama-3.1 8B in a few-shot setting**

```bash
# From the project root directory
uv run python -m scripts.benchmark \
    --dataset_dir data/blocks_4-sas \
    --num_blocks 4 \
    --model_type llama \
    --model_path "meta-llama/Llama-3.1-8B-Instruct" \
    --encoding_type sas \
    --output_dir ./benchmark_results \
    --max_plan_length 60 \
    --save_detailed_results \
    --llama_use_few_shot
```
