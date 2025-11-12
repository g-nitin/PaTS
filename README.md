# Planning as Time-Series (PaTS)

> [!NOTE]
> This is repository is a work in progress.

## Overview

This project introduces **Planning as Time-Series (PaTS)**, a framework that fundamentally shifts the paradigm of automated planning. Instead of casting planning as a search problem through state spaces or relying on explicit symbolic reasoning about preconditions and effects, PaTS reformulates it as a **time-series forecasting problem**.

The core idea is to learn an implicit transition function directly from expert-demonstrated plan trajectories. Given an initial state and a desired goal, a trained time-series model "forecasts" a sequence of intermediate states that collectively form a valid plan. This approach leverages the power of modern sequence modeling techniques to generate plans in a data-driven manner.

PaTS explores a diverse range of multivariate time-series models, including:

- **Recurrent Neural Networks (RNNs)** like LSTMs, which are well-suited for sequential data.
- **Modern compact MLP-based architectures** such as the Tiny Time Mixer (TTM), offering efficient forecasting.
- **Classical gradient-boosted models** like XGBoost, adapted for sequence prediction.
- **Large Language Models (LLMs)** like Llama, used for zero-shot or few-shot inference through prompt engineering.

To enable this, planning states are encoded as fixed-length numerical vectors. PaTS investigates two primary state representations:

- A **sparse binary predicate representation** (`bin`), similar to traditional PDDL ground predicates.
- A **compact position-based encoding** (`sas`), inspired by SAS+ formalisms, representing block positions.

By treating plan trajectories as multivariate time series, PaTS aims to learn the dynamics of state transitions, effectively bypassing explicit symbolic planning.

This repository provides the complete infrastructure to:

1.  **Generate Datasets**: Create planning problem datasets for domains like **Blocksworld** and **Grippers** with varying complexity.
2.  **Encode States**: Convert symbolic planning states into numerical vectors using either a sparse binary predicate representation (`bin`) or a compact position-based one (`sas`).
3.  **Train Models**: Train a diverse range of models, including LSTMs, modern MLP-based architectures like TTM, and classical gradient-boosted models like XGBoost.
4.  **Benchmark Performance**: Rigorously evaluate the generated plans for validity, correctness, and optimality using a dedicated benchmarking script.

## Project Structure

- `data/`: Contains scripts for generating Blocksworld planning problems, solving them with traditional planners (Fast Downward), extracting state trajectories, and encoding states into numerical formats. Also stores the raw and processed datasets.
- `scripts/`: Houses the core logic for training and benchmarking PaTS models. This includes PyTorch `Dataset` implementations, a generic `PlannableModel` interface, a `BlocksWorldValidator` for plan verification, and the main training/benchmarking scripts.
- `scripts/models/`: Contains the specific implementations and wrappers for each time-series model (LSTM, TTM, XGBoost, Llama) adapted for the PaTS framework.
- `pats.sh`: An example Slurm batch script for automating the training and benchmarking workflow on a cluster.

## Installation and Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-repo/PaTS.git
    cd PaTS
    ```
2.  **Set up Python environment**: We recommend using `uv` for dependency management.
    ```bash
    uv venv
    uv pip install -r requirements.txt
    ```
3.  **Install PDDL tools**: Ensure you have [`pddl-generators`](https://github.com/AI-Planning/pddl-generators/tree/main), [`Fast Downward`](https://github.com/aibasel/downward), and [`VAL`](https://github.com/KCL-Planning/VAL) installed and configured. The `data/generate_dataset_bw.sh` and `data/generate_dataset_gr.sh` scripts expect these tools to be accessible via the paths specified within the script. Update the `ROOT_DIR` variable in `data/generate_dataset.sh` to point to your planning tools installation.

## [Dataset Generation & Structure](data/README.md)

Details on how to generate data and the structure of the output can be found in the data directory's README.

## [Workflow & Scripts](scripts/README.md)

A detailed explanation of the workflow, key scripts, and evaluation metrics is available in the scripts directory's README.

## Running the System

All commands should be run from the project root directory. We use `uv run python -m` to ensure scripts are executed within the configured virtual environment.

**Note**: Before running, ensure your `data/generate_dataset_*.sh` scripts have the correct `ROOT_DIR`...

### 1. Generating the Dataset

First, configure the paths in `data/generate_dataset_bw.sh` (for Blocksworld) or `data/generate_dataset_gr.sh` (for Grippers). You can also set the desired encoding type (`bin` or `sas`). The script will create a directory like `data/blocks_4-sas/`.

```bash
# Navigate to the data directory
cd data


# Execute the script for Blocksworld with SAS+ encoding
# The script can be modified to change encoding type or other parameters.
./generate_dataset_bw.sh
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

**Example: Training an LSTM for Grippers with SAS+ encoding**

```bash
uv run python -m scripts.train_model \
 --model_type lstm \
 --domain grippers \
 --dataset_dir data/raw_problems/grippers/R2-O3-RM4 \
 --processed_block_encoding_dir data/processed_trajectories/grippers/R2-O3-RM4/sas \
 --num-robots 2 --num-objects 3 --num-rooms 4 \
 --encoding_type sas \
 --output_dir ./training_outputs \
 --epochs 400
```

### 3. Benchmarking a Trained Model

The `model_path` argument should point to the saved model artifact. For LSTM and XGBoost, this is typically a `.pth` or `.joblib` file. For TTM, it's the directory containing `model.pt` and `config.json`.

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
