## Workflow and Core Components

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

This directory contains the core logic for training, evaluating, and validating PaTS models.

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

The `benchmark.py` script provides a comprehensive evaluation of a model's planning capabilities. It computes a wide range of metrics, categorized as follows:

- **`solved_rate_strict`**: The percentage of problems where the generated plan is fully valid _and_ achieves the exact goal state. This is the primary measure of success.
- **`valid_sequence_rate`**: The percentage of plans that are physically and transitionally valid, regardless of goal achievement.
- **`goal_achievement_rate`**: The percentage of plans that end in the correct goal state, regardless of validity.
- **Plan Optimality Metrics**:
  - `avg_plan_length_ratio_for_solved`: The average ratio of the predicted plan length to the expert plan length, calculated only for strictly solved problems. A value close to 1 indicates optimality.
  - `avg_plan_length_diff_for_solved`: The average difference in plan length (predicted - expert) for strictly solved problems.
- **Partial Success Metrics**:
  - `avg_goal_jaccard_score` / `avg_goal_f1_score`: Measures the similarity between the final predicted state and the true goal state, useful for assessing partial goal satisfaction.
  - `avg_percent_physically_valid_states`: The average percentage of states within a generated plan that satisfy physical constraints.
  - `avg_percent_valid_transitions`: The average percentage of transitions within a generated plan that are legal.
- **Time-Series Metrics**: These metrics quantify the similarity between the _entire predicted trajectory_ and the expert trajectory.
  - `dtw_distance`: Dynamic Time Warping distance, a robust measure of similarity between two time series that may vary in speed or length.
  - `mean_per_step_hamming_dist`: The average Hamming distance (number of differing features) between predicted and expert states at each corresponding time step.
- **`violation_code_counts`**: A frequency count of different error types (e.g., `PHYS_BLOCK_FLOATING`, `TRANS_ILLEGAL_CHANGES`), useful for debugging and model analysis.

## Visualizations

The `benchmark.py` script automatically generates several plots to visualize the evaluation results, saved to the `--output_dir`:

- **Benchmark Summary**: Bar chart of key aggregated metrics (solved rate, valid rate, goal achievement, etc.).
- **Violation Distribution**: Bar chart showing the frequency of different plan violation types.
- **Plan Length Histograms**: Histograms comparing the distribution of predicted plan lengths against expert plan lengths.
- **Time-Series Metrics Histograms**: Histograms for DTW and Hamming distances.
- **Trajectory Comparison Plots**: For a limited number of solved problems, visual comparisons of predicted vs. expert state trajectories (heatmaps for binary, line plots for SAS+).
