## Workflow

The typical workflow for using PaTS is:

1.  **Dataset Generation**:

    - Configure and run `data/generate_dataset.sh` for the desired number of blocks and encoding type (`bin` or `sas`).
    - This populates a directory like `data/blocks_4-sas/` with all necessary files.

2.  **Model Training**:

    - Use the unified training script `scripts/train_model.py`.
    - Specify the `model_type` (`lstm`, `ttm`, `xgboost`), `encoding_type`, and other parameters.
    - Example (`lstm`, `sas`):
      ```bash
      uv run python -m scripts.train_model \
          --model_type lstm \
          --dataset_dir data/blocks_4-sas \
          --num_blocks 4 \
          --encoding_type sas \
          --output_dir ./training_outputs
      ```
    - The script saves model artifacts to a subdirectory within `--output_dir`.

3.  **Model Evaluation (Benchmarking)**:
    - Use `scripts/benchmark.py` for comprehensive evaluation on the test set.
    - This script works for all models, including inference-only models like Llama.
    - Example (evaluating a trained `lstm`):
      ```bash
      uv run python -m scripts.benchmark \
          --dataset_dir data/blocks_4-sas \
          --num_blocks 4 \
          --model_type lstm \
          --model_path ./training_outputs/lstm_N4/pats_lstm_model_N4.pth \
          --encoding_type sas \
          --output_dir ./benchmark_results
      ```
    - Example (evaluating `llama` in a few-shot setting):
      ```bash
      uv run python -m scripts.benchmark \
          --dataset_dir data/blocks_4-sas \
          --num_blocks 4 \
          --model_type llama \
          --model_path "meta-llama/Llama-3.1-8B-Instruct" \
          --encoding_type sas \
          --output_dir ./benchmark_results \
          --llama_use_few_shot
      ```

## Key Scripts and Components

- **`train_model.py`**: The **central script for training** PaTS models (LSTM, TTM, XGBoost). It handles dataset loading, model instantiation, and training loops.
- **`benchmark.py`**: The **central script for evaluating** all trained or pre-trained PaTS models. It uses a standardized `PlannableModel` interface to interact with different models, generates plans for test problems, and computes a rich set of performance metrics.
- **`pats_dataset.py`**: A unified PyTorch `Dataset` class that loads pre-encoded trajectories and goals based on the specified encoding type.
- **`BlocksWorldValidator.py`**: A crucial class for checking the physical validity of states and the legality of transitions. It dynamically adapts to the encoding (`bin` or `sas`) by reading the `encoding_info.json` file from the dataset directory.
- **`PlannableModel.py`**: An abstract base class that defines the standard interface (`load_model`, `predict_sequence`) for all models, enabling the benchmark script to treat them interchangeably.
- **`models/`**: This directory contains the specific implementations and wrappers for each model:
  - `lstm.py`: The `PaTS_LSTM` model, which supports both binary and SAS+ (with an embedding layer) encodings.
  - `ttm.py`: The `BlocksWorldTTM` wrapper for the TTM forecasting model.
  - `xgboost.py`: The `XGBoostPlanner` implementation.
  - `llama.py`: The `LlamaWrapper` for zero-shot and few-shot inference with large language models.

## Benchmarking and Evaluation Metrics

The `benchmark.py` script computes a wide range of metrics to provide a holistic view of a model's planning capabilities, including:

- **`solved_rate_strict`**: The percentage of problems where the generated plan is fully valid _and_ achieves the exact goal state. This is the primary measure of success.
- **`valid_sequence_rate`**: The percentage of plans that are physically and transitionally valid, regardless of goal achievement.
- **`goal_achievement_rate`**: The percentage of plans that end in the correct goal state, regardless of validity.
- **Plan Optimality Metrics**: Compares the length of solved plans to expert plans (`avg_plan_length_ratio_for_solved`).
- **Partial Success Metrics**: Measures partial goal satisfaction (`avg_goal_f1_score`) and the percentage of valid states/transitions within a plan.
- **Time-Series Metrics**: `dtw_distance` and `mean_per_step_hamming_dist` measure the similarity of the predicted trajectory to the expert trajectory.
- **`violation_code_counts`**: A frequency count of different error types, useful for debugging and model analysis.
