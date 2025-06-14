## Workflow

The typical workflow for using PaTS is:

1.  **Dataset Generation**:

    - Configure `data/generate_dataset.sh` (paths, number of blocks, problems per config).
    - Run the script: `cd data && ./generate_dataset.sh`.
    - This will populate `data/blocks_<N>/` directories with PDDL files, plans, VAL logs, encoded trajectories, and the predicate manifest.

2.  **Model Training**:

    - Choose a model script (e.g., `scripts/models/ttm.py` or `scripts/models/lstm.py`).
    - Configure training parameters within the script or via its command-line arguments (e.g., dataset path, output directory, hyperparameters).
    - Run the training script. For example:
      ```bash
      # For TTM
      uv run python scripts/models/ttm.py --dataset_dir data/blocks_4 --dataset_split_dir data/blocks_4 --num_blocks--num_epochs 100
      # For LSTM
      uv run python scripts/models/lstm.py data/blocks_4 output_lstm_N4 --num_blocks 4 --epochs 100
      ```
    - The script will load data based on the `train_files.txt` and `val_files.txt` (or its own splitting logic), train the model, and save the trained model weights and configuration.

3.  **Model Evaluation (Benchmarking)**:
    - Use the `scripts/benchmark.py` script for comprehensive evaluation.
    - This script requires the path to the dataset, the number of blocks, model type, path to the trained model, and an output directory.
    - Example:
      ```bash
      uv run python scripts/benchmark.py \
          --dataset_dir ./data \
          --num_blocks 4 \
          --model_type ttm \
          --model_path ./output_ttm_N4/final_model_assets \
          --output_dir ./benchmark_results_ttm_N4
      ```
    - The script loads test problems specified in `data/blocks_<N>/test_files.txt`, uses the appropriate model wrapper to generate plans, validates them using `BlocksWorldValidator` (which uses the predicate manifest), and computes a range of metrics.

## Key Scripts and Components

- **`data/generate_dataset.sh`**: Automates the entire data generation pipeline from PDDL problem generation to encoded trajectories and predicate manifests.
- **`data/parse_and_encode.py`**: Parses PDDL files (for initial/goal states) and VAL output logs (for state transitions), reconstructs state trajectories, encodes them into binary vectors based on a generated predicate order, and saves the binary data along with the crucial `predicate_manifest_<N>.txt` file.
- **`data/analyze_dataset_splits.py`**: Analyzes the generated dataset for distributions and splits it into training, validation, and test sets, creating `*_files.txt`.
- **`scripts/BlocksWorldValidator.py`**:
  - Contains the `BlocksWorldValidator` class responsible for checking the physical validity of individual states and the legality of transitions between states in the Blocks World domain.
  - **Crucially, it is initialized with `num_blocks` and the path to the `predicate_manifest_<N>.txt` file.** This allows it to dynamically understand the structure of the binary state vectors it receives, making it robust to encoding changes.
  - It defines `Violation` and `ValidationResult` dataclasses to structure validation output.
- **`scripts/benchmark.py`**:
  - The **central script for evaluating trained PaTS models.**
  - It uses an abstract `PlannableModel` class and specific wrappers (e.g., `TTMWrapper`, `LSTMWrapper`) to interact with different models in a standardized way.
  - Loads test data (initial states, goal states, expert trajectories) based on `test_files.txt`.
  - For each test problem, it instructs the loaded model to generate a plan.
  - Passes the generated plan and goal state to an instance of `BlocksWorldValidator` (configured with the correct predicate manifest) for validation.
  - Collects detailed `ValidationResult` objects for each problem.
  - Computes and outputs aggregated performance metrics.
- **`scripts/models/ttm.py`**: Implements training and prediction logic for the Tiny Time Mixer (TTM) model. It includes the `BlocksWorldDataset` for TTM-specific data loading (using `.npy` files, split files like `train_files.txt`, and the `predicate_manifest_<N>.txt`) and the `BlocksWorldTTM` class to manage the model. Its main focus is now on training and saving models.
- **`scripts/models/lstm.py`**: Implements training and prediction logic for a baseline LSTM model. Its main focus is now on training and saving models.
- **`scripts/models/plansformer.py`**: (Placeholder) Intended for a Transformer-based planning model.

## Benchmarking and Evaluation (`scripts/benchmark.py`)

The `benchmark.py` script provides a standardized and extensible way to evaluate different PaTS models.

### Validator

The `BlocksWorldValidator` is key to assessing plan quality.

- It checks for physical consistency (e.g., no block floating, a block isn't on two things at once).
- It checks for legal transitions (e.g., only a valid Blocks World action could have occurred).
- It uses the `predicate_manifest_<N>.txt` file to interpret the binary state vectors, ensuring it aligns with the current encoding scheme.

### Model Wrappers

To allow `benchmark.py` to work with various models, it uses an adapter pattern:

- `PlannableModel` (Abstract Base Class): Defines the interface a model must implement (`load_model`, `predict_sequence`, `model_name`, `state_dim`).
- `TTMWrapper`, `LSTMWrapper`: Concrete implementations for TTM and LSTM, adapting their specific loading and prediction methods to the `PlannableModel` interface. New models can be benchmarked by creating a similar wrapper.

### Metrics

`benchmark.py` computes a range of metrics, including:

- **`num_samples`**: Total test problems evaluated.
- **`valid_sequence_rate`**: Percentage of generated plans that are fully valid according to the `BlocksWorldValidator` (all states and transitions are valid).
- **`goal_achievement_rate`**: Percentage of generated plans where the final state exactly matches the goal state.
- **`solved_rate_strict`**: Percentage of problems where the generated plan is both fully valid AND achieves the exact goal state. This is a key indicator of overall planning success.
- **`avg_predicted_plan_length`**: Average length of the generated plans.
- **`avg_expert_plan_length`**: Average length of the expert-generated plans for the same problems.
- **`avg_goal_jaccard_score`**: Average Jaccard index between the true predicates in the predicted final state and the goal state. Measures partial goal satisfaction.
- **`avg_goal_f1_score`**: Average F1-score for predicate match in the final state vs. goal state.
- **`avg_percent_physically_valid_states`**: Average percentage of states within each generated plan that are individually physically valid.
- **`avg_percent_valid_transitions`**: Average percentage of transitions within each generated plan that are legal.
- **`avg_plan_length_ratio_for_solved`**: For strictly solved problems, the average ratio of `predicted_plan_length / expert_plan_length`.
- **`avg_plan_length_diff_for_solved`**: For strictly solved problems, the average difference `predicted_plan_length - expert_plan_length`.
- **`violation_code_counts`**: A frequency count of different types of validation errors (e.g., `PHYS_BLOCK_FLOATING`, `TRANS_ILLEGAL_CHANGES`) encountered across all invalid plans.

These metrics provide a multi-faceted view of the model's planning capabilities.
