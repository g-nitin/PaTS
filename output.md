- `PaTS/README.md`
````md
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

````

- `PaTS/data/README.md`
````md
## Dataset

The dataset for PaTS consists of solved planning problem instances from the chosen domain (e.g., Blocksworld).

### Generation Process

The dataset is generated using the `data/generate_dataset.sh` script. This script automates:

1.  **Problem Generation**: PDDL problem files (`.pddl`) are created.
2.  **Plan Generation**: Fast Downward finds a solution plan (`.plan`).
3.  **Plan Validation & State Extraction**: VAL validates the plan and its verbose output (`.val.log`) details state changes.
4.  **Parsing and Encoding**: `data/parse_and_encode.py` processes PDDL files and VAL logs. It is controlled by the `--encoding_type` flag (`bin` or `sas`). It reconstructs the state trajectory, encodes each state into the chosen vector format, and saves the encoded data.
5.  **Dataset Splitting**: `data/analyze_dataset_splits.py` creates `train_files.txt`, `val_files.txt`, `test_files.txt`.

### Data Structure and Format

All generated data for `N` blocks is organized within `data/blocks_<N>/`. Key files per problem instance `blocks_<N>_problem_<M>`:

- `pddl/blocks_<N>_problem_<M>.pddl`: PDDL problem.
- `plans/blocks_<N>_problem_<M>.plan`: Expert plan.
- `val_out/blocks_<N>_problem_<M>.val.log`: VAL output.
- `trajectories_text/blocks_<N>_problem_<M>.traj.txt`: Human-readable trajectory.
- `trajectories_bin/blocks_<N>_problem_<M>.traj.<encoding>.npy`: NumPy array `(L, F)` of encoded states (e.g., `.traj.bin.npy` or `.traj.sas.npy`).
- `trajectories_bin/blocks_<N>_problem_<M>.goal.<encoding>.npy`: NumPy array `(F,)` of the encoded goal state.
- `encoding_info_<N>.json`: **Crucial file** describing the encoding used for this dataset (type, feature dimension, and path to manifest if applicable).
- `predicate_manifest_<N>.txt`: For `bin` encoding, this lists all predicates in order, defining the feature map.
- `train_files.txt`, `val_files.txt`, `test_files.txt`: Lists of problem basenames for each split.

### State Encoding

PaTS supports multiple state encoding schemes, controlled by the `--encoding_type` flag in `parse_and_encode.py`.

#### Binary Predicate Encoding (`--encoding_type bin`)

- **Representation**: A long binary vector where each element corresponds to a specific ground predicate (e.g., `(on-table b1)`, `(on b1 b2)`). `1` means true, `0` means false.
- **Size**: Scales quadratically with the number of blocks, O(n²).
- **Configuration**: This encoding is defined by a `predicate_manifest_<N>.txt` file, which lists every possible predicate in a fixed order. The `encoding_info_<N>.json` file will point to this manifest.

#### SAS+ Position Vector Encoding (`--encoding_type sas`)

- **Representation**: A compact integer vector where the _index_ represents the block and the _value_ represents its position.
  - `vector[i]` corresponds to block `b(i+1)`.
  - Value `0`: The block is on the table.
  - Value `j > 0`: The block is on top of block `bj`.
  - Value `-1`: The block is being held by the arm.
- **Example (4 blocks)**: The state "A on B, B on table, C on D, D on table" (with A=b1, B=b2, C=b3, D=b4) is encoded as `[2, 0, 4, 0]`.
- **Size**: Scales linearly with the number of blocks, O(n).
- **Configuration**: The `encoding_info_<N>.json` file will specify the type as `sas` and list the block order. No separate manifest file is needed.

### Encoding Information (`encoding_info_<N>.json`)

This JSON file is the primary source of truth for how states are encoded for a given `num_blocks`. It is read by the `BlocksWorldValidator` and other components to adapt to the current scheme.

````

- `PaTS/scripts/README.md`
````md
## Workflow

The typical workflow for using PaTS is:

1.  **Dataset Generation**:

    - Configure `data/generate_dataset.sh` (paths, number of blocks, problems per config).
    - Run the script: `cd data && ./generate_dataset.sh`.
    - This will populate `data/blocks_<N>/` directories with PDDL files, plans, VAL logs, encoded trajectories, and the predicate manifest.

2.  **Model Training**:

    - Use the unified training script `scripts/train_model.py`.
    - Specify the `model_type` (e.g., `ttm`, `lstm`) and other relevant parameters.
    - Example:

      ```bash
      # For TTM
      uv run python -m scripts.train_model \
          --model_type ttm \
          --dataset_dir data/blocks_4 \
          --dataset_split_dir data/blocks_4 \
          --num_blocks 4 \
          --output_dir ./training_outputs \
          --epochs 100

      # For LSTM
      uv run python -m scripts.train_model \
          --model_type lstm \
          --dataset_dir data/blocks_4 \
          --dataset_split_dir data/blocks_4 \
          --num_blocks 4 \
          --encoding_type bin \ # Specify 'bin' or 'sas'
          --output_dir ./training_outputs \
          --epochs 200
      ```

    - The script loads data based on `train_files.txt` and `val_files.txt`, trains the specified model, and saves weights/configuration to a subdirectory within `--output_dir` (e.g., `./training_outputs/ttm_N4/`).

3.  **Model Evaluation (Benchmarking)**:
    - Use the `scripts/benchmark.py` script for comprehensive evaluation.
    - This script requires the path to the dataset, the number of blocks, model type, path to the trained model, and an output directory.
    - Example:
      ```bash
      uv run python -m scripts.benchmark.py \
          --dataset_dir data/blocks_4 \
          --num_blocks 4 \
          --model_type ttm \ # or lstm
          --model_path ./training_outputs/ttm_N4/final_model_assets \ # Adjust path based on actual saved model
      ```

-          --encoding_type bin \ # Must match the model's training encoding
          --output_dir ./benchmark_results_ttm_N4
      ```
  - The script loads test problems specified in `data/blocks_<N>/test_files.txt`, uses the appropriate model wrapper to generate plans, validates them using `BlocksWorldValidator` (which uses the predicate manifest), and computes a range of metrics.

## Key Scripts and Components

- **`data/generate_dataset.sh`**: Automates the entire data generation pipeline from PDDL problem generation to encoded trajectories and predicate manifests.
- **`data/parse_and_encode.py`**: Parses PDDL files (for initial/goal states) and VAL output logs (for state transitions), reconstructs state trajectories, encodes them into binary vectors based on a generated predicate order, and saves the binary data along with the crucial `predicate_manifest_<N>.txt` file.
- **`data/analyze_dataset_splits.py`**: Analyzes the generated dataset for distributions and splits it into training, validation, and test sets, creating `*_files.txt`.
- **`scripts/train_model.py`**: The **central script for training PaTS models (LSTM, TTM, etc.)**. It handles dataset loading, model instantiation, and invoking model-specific training loops. It can optionally add a constraint violation loss term during training (for LSTM) by using the `--use_constraint_loss` flag. This leverages the `BlocksWorldValidator` to guide the model towards generating physically valid states.
- **`scripts/pats_dataset.py`**: Contains the `PaTSDataset` class, a unified PyTorch Dataset for loading pre-encoded binary trajectories and goal states from `.npy` files based on split file lists (e.g., `train_files.txt`).
- **`scripts/BlocksWorldValidator.py`**:
  - Contains the `BlocksWorldValidator` class responsible for checking the physical validity of individual states and the legality of transitions between states in the Blocks World domain.
  - **Crucially, it is initialized with `num_blocks` and an `encoding_type` ('bin' or 'sas').** For binary encoding, it also requires the path to the `predicate_manifest_<N>.txt` file. This allows it to dynamically apply the correct validation rules for the given state representation.
  - Provides a differentiable `calculate_constraint_violation_loss` method that can be used during training to penalize the model for generating physically invalid states, directly embedding domain rules into the learning process.
  - It defines `Violation` and `ValidationResult` dataclasses to structure validation output.
- **`scripts/benchmark.py`**:
  - The **central script for evaluating trained PaTS models.**
  - It uses an abstract `PlannableModel` class and specific wrappers (e.g., `TTMWrapper`, `LSTMWrapper`) to interact with different models in a standardized way.
  - Loads test data (initial states, goal states, expert trajectories) based on `test_files.txt`.
  - For each test problem, it instructs the loaded model to generate a plan.
  - Passes the generated plan and goal state to an instance of `BlocksWorldValidator` (configured with the correct encoding type) for validation.
  - Collects detailed `ValidationResult` objects for each problem.
  - Computes and outputs aggregated performance metrics.
- **`scripts/models/ttm.py`**: Contains the `BlocksWorldTTM` class (managing TTM training and prediction), `TTMDataCollator`, and related utilities. Training is orchestrated by `scripts/train_model.py`.
- **`scripts/models/lstm.py`**: Contains the `PaTS_LSTM` model definition (an `nn.Module`) and the `lstm_collate_fn`. Training is orchestrated by `scripts/train_model.py`.
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

````

- `PaTS/scripts/benchmark.py`
```py
import argparse
import json
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from fastdtw import fastdtw
from scipy.spatial.distance import hamming

from .BlocksWorldValidator import BlocksWorldValidator, ValidationResult
from .models.lstm import PaTS_LSTM
from .models.ttm import BlocksWorldTTM
from .models.ttm import ModelConfig as TTMModelConfig

# Setup device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Benchmarking using device: {DEVICE}")


# ** Abstract Plannable Model **
class PlannableModel(ABC):
    def __init__(self, model_path: Path, num_blocks: int, device: torch.device):
        self.model_path = model_path
        self.num_blocks = num_blocks
        self.device = device
        self.model: Any = None  # To be initialized by load_model

    @abstractmethod
    def load_model(self):
        """Loads the model from self.model_path."""
        pass

    @abstractmethod
    def predict_sequence(self, initial_state_np: np.ndarray, goal_state_np: np.ndarray, max_length: int) -> List[List[int]]:
        """
        Predicts a sequence of states (plan).
        Inputs are 0/1 numpy arrays. Output is List of 0/1 state lists.
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Returns a descriptive name of the model."""
        pass

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Returns the feature dimension of the states the model expects/produces."""
        pass


# ** TTM Wrapper **
class TTMWrapper(PlannableModel):
    def __init__(self, model_path: Path, num_blocks: int, device: torch.device):
        super().__init__(model_path, num_blocks, device)
        self.ttm_instance: BlocksWorldTTM
        self.config: TTMModelConfig

    def load_model(self):
        print(f"Loading TTM model from: {self.model_path}")
        self.ttm_instance = BlocksWorldTTM.load(self.model_path, self.device)
        self.model = self.ttm_instance.model  # The actual torch model
        self.config = self.ttm_instance.config
        print(
            f"TTM model loaded. Context: {self.config.context_length}, Pred Len: {self.config.prediction_length}, State Dim: {self.config.state_dim}"
        )

    def predict_sequence(self, initial_state_np: np.ndarray, goal_state_np: np.ndarray, max_length: int) -> List[List[int]]:
        if self.model is None or self.config is None:
            raise RuntimeError("TTM model not loaded. Call load_model() first.")

        self.ttm_instance.model.eval()
        with torch.no_grad():
            # Convert numpy inputs to torch tensors and MOVE TO THE CORRECT DEVICE
            initial_state_tensor = torch.from_numpy(initial_state_np.astype(np.float32)).unsqueeze(0).to(self.device)
            goal_state_tensor = torch.from_numpy(goal_state_np.astype(np.float32)).unsqueeze(0).to(self.device)

            # Initialize the context by repeating the initial state
            # This will now be on the correct device because initial_state_tensor is
            current_context = initial_state_tensor.unsqueeze(1).repeat(1, self.config.context_length, 1)

            # The plan starts with the initial state
            generated_plan_tensors = [initial_state_tensor.squeeze(0)]

            for step in range(max_length - 1):  # max_length includes S0
                # Predict the next sequence of states from the current context
                # The underlying predict method now takes the full context
                predicted_future_sequence = self.ttm_instance.predict(current_context, goal_state_tensor)  # (1, P, F)

                # We only need the very next state for our autoregressive step
                next_state = predicted_future_sequence[:, 0, :]  # (1, F)

                # Stagnation check: if the model predicts no change, stop.
                # This prevents infinitely long plans of repeating states.
                last_state_in_context = current_context[:, -1, :]
                if torch.equal(next_state, last_state_in_context):
                    # print(f"TTM: Stagnation detected at step {step + 1}. Stopping.")
                    break

                generated_plan_tensors.append(next_state.squeeze(0))

                # Update the context for the next iteration:
                # Roll the context to the left and append the new state at the end
                current_context = torch.roll(current_context, shifts=-1, dims=1)
                current_context[:, -1, :] = next_state

                # Goal achievement check
                if torch.equal(next_state.squeeze(0), goal_state_tensor.squeeze(0)):
                    # print(f"TTM: Goal reached at step {step + 1}.")
                    break

            # Convert the final list of tensors to the required List[List[int]] format
            final_plan_np = torch.stack(generated_plan_tensors).cpu().numpy()
            return [state.astype(int).tolist() for state in final_plan_np]

    @property
    def model_name(self) -> str:
        return f"TTM_{Path(self.config.ttm_model_path).name}" if self.config else "TTM_unloaded"

    @property
    def state_dim(self) -> int:
        if not self.config or self.config.state_dim is None:
            raise ValueError("TTM config not loaded or state_dim not set.")
        return self.config.state_dim


# ** LSTM Wrapper **
class LSTMWrapper(PlannableModel):
    def __init__(self, model_path: Path, num_blocks: int, device: torch.device):
        super().__init__(model_path, num_blocks, device)
        self.lstm_model: PaTS_LSTM
        self._state_dim: Optional[int] = None  # Will be loaded from checkpoint

    def load_model(self):
        if not self.model_path.is_file():
            raise FileNotFoundError(f"LSTM model file not found: {self.model_path}")

        print(f"Loading LSTM model from: {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Read model parameters from the checkpoint
        num_features = checkpoint.get("num_features")
        hidden_size = checkpoint.get("hidden_size")
        num_lstm_layers = checkpoint.get("num_lstm_layers")
        dropout_prob = checkpoint.get("dropout_prob", 0.2)

        # Check if the model was trained with the MLM task
        was_trained_with_mlm = checkpoint.get("use_mlm_task", False)

        encoding_type = checkpoint.get("encoding_type", "bin")  # Default to 'bin' if not found
        num_blocks = checkpoint.get("target_num_blocks")
        embedding_dim = checkpoint.get("embedding_dim")

        if not all([num_features, hidden_size, num_lstm_layers, num_blocks is not None]):
            raise ValueError(
                "LSTM checkpoint missing required parameters (num_features, hidden_size, num_lstm_layers, target_num_blocks)."
            )

        self._state_dim = num_features
        # Instantiate the model with the correct configuration based on the checkpoint
        self.lstm_model = PaTS_LSTM(
            num_features=num_features,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
            dropout_prob=dropout_prob,
            use_mlm_task=was_trained_with_mlm,
            encoding_type=encoding_type,
            num_blocks=num_blocks,
            embedding_dim=embedding_dim,
        ).to(self.device)

        self.lstm_model.load_state_dict(checkpoint["model_state_dict"])
        self.lstm_model.eval()
        self.model = self.lstm_model  # For consistency with PlannableModel
        print(
            f"LSTM model loaded. Features: {num_features}, Hidden: {hidden_size}, "
            f"Layers: {num_lstm_layers}, Encoding: {encoding_type}"
        )

    def predict_sequence(self, initial_state_np: np.ndarray, goal_state_np: np.ndarray, max_length: int) -> List[List[int]]:
        if self.lstm_model is None:
            raise RuntimeError("LSTM model not loaded. Call load_model() first.")

        # generate_plan_lstm is a standalone function in lstm.py, let's adapt its core logic
        self.lstm_model.eval()
        with torch.no_grad():
            # Create tensors with the correct dtype based on the model's encoding type
            if self.lstm_model.encoding_type == "sas":
                # For SAS+, the input is integer indices and requires a LongTensor
                # Safeguard exists to clamp out-of-bounds indices from input files
                # This mirrors the safeguard in train_model.py and prevents CUDA asserts.
                # The valid indices are 0 (table) through N (block N), where N = num_blocks.
                max_valid_index = self.lstm_model.num_blocks
                initial_state_np.clip(min=0, max=max_valid_index, out=initial_state_np)
                goal_state_np.clip(min=0, max=max_valid_index, out=goal_state_np)
                current_S_tensor = torch.LongTensor(initial_state_np).unsqueeze(0).to(self.device)
                goal_S_tensor = torch.LongTensor(goal_state_np).unsqueeze(0).to(self.device)
            else:  # 'bin' encoding
                current_S_tensor = torch.FloatTensor(initial_state_np).unsqueeze(0).to(self.device)
                goal_S_tensor = torch.FloatTensor(goal_state_np).unsqueeze(0).to(self.device)

            h_prev = torch.zeros(self.lstm_model.num_lstm_layers, 1, self.lstm_model.hidden_size).to(self.device)
            c_prev = torch.zeros(self.lstm_model.num_lstm_layers, 1, self.lstm_model.hidden_size).to(self.device)

            generated_plan_tensors = [current_S_tensor.clone()]  # Start with S0

            for step in range(max_length - 1):  # Max_length includes S0
                next_S, _, h_next, c_next = self.lstm_model.predict_step(current_S_tensor, goal_S_tensor, h_prev, c_prev)
                generated_plan_tensors.append(next_S.clone())
                current_S_tensor = next_S
                h_prev, c_prev = h_next, c_next

                if torch.equal(current_S_tensor.squeeze(), goal_S_tensor.squeeze()):  # Compare 1D tensors
                    # print(f"LSTM: Goal reached at step {step + 1}.")
                    break

            final_plan_np = torch.cat(generated_plan_tensors, dim=0).cpu().numpy()
            return [state.astype(int).tolist() for state in final_plan_np]

    @property
    def model_name(self) -> str:
        return "PaTS_LSTM"

    @property
    def state_dim(self) -> int:
        if self._state_dim is None:
            raise ValueError("LSTM model not loaded or state_dim not set.")
        return self._state_dim


def compute_timeseries_metrics(
    predicted_plan: List[List[int]], expert_plan: np.ndarray, encoding_type: str
) -> Dict[str, float]:
    """
    Computes time-series specific metrics. Jaccard is only computed for 'bin' encoding.

    :param predicted_plan: The plan generated by the model, as a list of binary state lists.
    :param expert_plan: The expert plan from the dataset, as a NumPy array.
    :return: A dictionary containing the calculated metrics. Returns -1.0 for metrics that cannot be computed.
    """

    # Initial validation
    if not predicted_plan or expert_plan.shape[0] == 0:
        return {
            "mean_per_step_hamming_dist": -1.0,
            "mean_per_step_jaccard": -1.0,
            "dtw_distance": -1.0,
        }

    predicted_plan_np = np.array(predicted_plan, dtype=np.int8)

    # 1. Per-step metrics (Hamming, Jaccard)
    min_len = min(predicted_plan_np.shape[0], expert_plan.shape[0])
    step_hammings = []
    step_jaccards = []

    if min_len > 0:
        for i in range(min_len):
            pred_state = predicted_plan_np[i]
            expert_state = expert_plan[i]

            # Hamming Distance (raw count of differing bits)
            step_hammings.append(np.sum(pred_state != expert_state))

    mean_hamming = np.mean(step_hammings) if step_hammings else -1.0

    # 2. Dynamic Time Warping (DTW) using fastdtw
    # fastdtw is generally robust, but we avoid for empty sequences.
    if predicted_plan_np.shape[0] < 1 or expert_plan.shape[0] < 1:
        dtw_dist = -1.0
    else:
        try:
            # Use fastdtw and explicitly provide the distance function.
            # `scipy.spatial.distance.hamming` calculates the *proportion* of
            # differing elements. We multiply by the number of features to get the
            # raw count, which is a more intuitive DTW cost.
            num_features = predicted_plan_np.shape[1]

            # Define a lambda to wrap the hamming distance and scale it
            hamming_dist_raw = lambda u, v: hamming(u, v) * num_features

            # The fastdtw function is the robust way to do this.
            dtw_dist, path = fastdtw(predicted_plan_np, expert_plan, dist=hamming_dist_raw)

        except Exception as e:
            print(f"\n[Warning] DTW calculation with fastdtw failed unexpectedly. Error: {e}")
            dtw_dist = -1.0

    mean_jaccard = -1.0
    if encoding_type == "bin":
        if min_len > 0:
            for i in range(min_len):
                pred_state = predicted_plan_np[i]
                expert_state = expert_plan[i]
                pred_true_indices = set(np.where(pred_state == 1)[0])
                expert_true_indices = set(np.where(expert_state == 1)[0])
                intersection_len = len(pred_true_indices.intersection(expert_true_indices))
                union_len = len(pred_true_indices.union(expert_true_indices))
                jaccard = intersection_len / union_len if union_len > 0 else 1.0
                step_jaccards.append(jaccard)
        mean_jaccard = np.mean(step_jaccards) if step_jaccards else -1.0

    return {
        "mean_per_step_hamming_dist": float(mean_hamming),
        "mean_per_step_jaccard": float(mean_jaccard),
        "dtw_distance": float(dtw_dist),
    }


def get_plannable_model(model_type: str, model_path: Path, num_blocks: int, device: torch.device) -> PlannableModel:
    """Factory function to get a PlannableModel instance."""
    model_type_lower = model_type.lower()
    if model_type_lower == "ttm":
        return TTMWrapper(model_path, num_blocks, device)
    elif model_type_lower == "lstm":
        return LSTMWrapper(model_path, num_blocks, device)
    # elif model_type_lower == "plansformer":
    #     # return PlansformerWrapper(model_path, num_blocks, device) # Placeholder
    #     raise NotImplementedError("Plansformer wrapper not yet implemented.")
    else:
        raise ValueError(f"Unsupported model type for PlannableModel factory: {model_type}")


# ** Helper Functions for Benchmarking **
def load_problem_basenames_from_split_file(split_file_path: Path) -> List[str]:
    if not split_file_path.exists():
        print(f"Warning: Split file not found: {split_file_path}")
        return []
    with open(split_file_path, "r") as f:
        basenames = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return basenames


def compute_aggregated_metrics(
    results: List[ValidationResult], expert_plan_lengths: List[int], timeseries_metrics_list: List[Dict[str, float]]
) -> Dict[str, Any]:
    num_samples = len(results)
    if num_samples == 0:
        return {"message": "No results to aggregate."}

    agg_metrics = {
        "num_samples": num_samples,
        "valid_sequence_rate": 0.0,  # Plan is fully valid by validator rules
        "goal_achievement_rate": 0.0,  # Final state is goal state
        "solved_rate_strict": 0.0,  # Valid sequence AND final state is goal
        "avg_predicted_plan_length": 0.0,
        "avg_expert_plan_length": np.mean(expert_plan_lengths) if expert_plan_lengths else 0.0,
        "avg_goal_jaccard_score": 0.0,
        "avg_goal_f1_score": 0.0,
        "avg_percent_physically_valid_states": 0.0,
        "avg_percent_valid_transitions": 0.0,
        "avg_plan_length_ratio_for_solved": 0.0,  # For strictly solved problems
        "avg_plan_length_diff_for_solved": 0.0,  # For strictly solved problems
        "avg_mean_per_step_hamming_dist": 0.0,
        "avg_mean_per_step_jaccard": 0.0,
        "avg_dtw_distance": 0.0,
        "violation_code_counts": Counter(),
    }

    valid_sequences_count = 0
    goal_achieved_count = 0
    solved_strict_count = 0
    total_predicted_plan_length = 0
    total_jaccard = 0.0
    total_f1 = 0.0
    total_perc_phys_valid_states = 0.0
    total_perc_valid_transitions = 0.0

    solved_plan_length_ratios = []
    solved_plan_length_diffs = []

    total_hamming = 0.0
    total_ts_jaccard = 0.0
    total_dtw = 0.0
    valid_ts_metrics_count = 0

    for ts_metrics in timeseries_metrics_list:
        if ts_metrics and ts_metrics.get("dtw_distance", -1.0) >= 0:
            total_hamming += ts_metrics["mean_per_step_hamming_dist"]
            total_ts_jaccard += ts_metrics["mean_per_step_jaccard"]
            total_dtw += ts_metrics["dtw_distance"]
            valid_ts_metrics_count += 1

    for i, res in enumerate(results):
        if res.is_valid:
            valid_sequences_count += 1
        if res.metrics.get("goal_achievement", 0.0) == 1.0:
            goal_achieved_count += 1

        is_strictly_solved = res.is_valid and res.metrics.get("goal_achievement", 0.0) == 1.0
        if is_strictly_solved:
            solved_strict_count += 1
            if expert_plan_lengths and i < len(expert_plan_lengths) and expert_plan_lengths[i] > 0:
                ratio = res.predicted_plan_length / expert_plan_lengths[i]
                diff = res.predicted_plan_length - expert_plan_lengths[i]
                solved_plan_length_ratios.append(ratio)
                solved_plan_length_diffs.append(diff)

        total_predicted_plan_length += res.predicted_plan_length
        total_jaccard += res.goal_jaccard_score
        total_f1 += res.goal_f1_score
        total_perc_phys_valid_states += res.metrics.get("percent_physically_valid_states", 0.0)
        total_perc_valid_transitions += res.metrics.get("percent_valid_transitions", 0.0)

        for violation in res.violations:
            agg_metrics["violation_code_counts"][violation.code] += 1

    agg_metrics["valid_sequence_rate"] = valid_sequences_count / num_samples
    agg_metrics["goal_achievement_rate"] = goal_achieved_count / num_samples
    agg_metrics["solved_rate_strict"] = solved_strict_count / num_samples
    agg_metrics["avg_predicted_plan_length"] = total_predicted_plan_length / num_samples
    agg_metrics["avg_goal_jaccard_score"] = total_jaccard / num_samples
    agg_metrics["avg_goal_f1_score"] = total_f1 / num_samples
    agg_metrics["avg_percent_physically_valid_states"] = total_perc_phys_valid_states / num_samples
    agg_metrics["avg_percent_valid_transitions"] = total_perc_valid_transitions / num_samples

    if valid_ts_metrics_count > 0:
        agg_metrics["avg_mean_per_step_hamming_dist"] = total_hamming / valid_ts_metrics_count
        agg_metrics["avg_mean_per_step_jaccard"] = total_ts_jaccard / valid_ts_metrics_count
        agg_metrics["avg_dtw_distance"] = total_dtw / valid_ts_metrics_count

    if solved_plan_length_ratios:
        agg_metrics["avg_plan_length_ratio_for_solved"] = np.mean(solved_plan_length_ratios)
        agg_metrics["avg_plan_length_diff_for_solved"] = np.mean(solved_plan_length_diffs)

    # Convert Counter to dict for JSON serialization
    agg_metrics["violation_code_counts"] = dict(agg_metrics["violation_code_counts"])

    return agg_metrics


# ** Main Benchmarking Logic **
def run_benchmark(args: argparse.Namespace):
    data_root_dir = Path(args.dataset_dir)
    num_blocks = args.num_blocks
    model_type = args.model_type
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Construct paths based on num_blocks
    if not data_root_dir.exists():
        print(f"ERROR: Block-specific data directory not found: {data_root_dir}")
        return

    test_split_file = data_root_dir / "test_files.txt"
    problem_basenames = load_problem_basenames_from_split_file(test_split_file)
    if not problem_basenames:
        print(f"No test problems found in {test_split_file}. Exiting.")
        return

    print(f"Found {len(problem_basenames)} test problems for N={num_blocks}.")

    # Initialize Validator
    print(f"Initializing validator for N={num_blocks} with '{args.encoding_type}' encoding.")

    validator = None
    if args.encoding_type == "bin":
        # Construct the path to the predicate manifest for the specific num_blocks
        predicate_manifest_file = data_root_dir / f"predicate_manifest_{num_blocks}.txt"
        if not predicate_manifest_file.exists():
            print(f"ERROR: Predicate manifest file not found: {predicate_manifest_file}")
            return
        try:
            validator = BlocksWorldValidator(
                num_blocks, args.encoding_type, predicate_manifest_file=predicate_manifest_file
            )
            print(f"Validator initialized with state_size={validator.state_size}.")
        except Exception as e:
            print(f"ERROR: Failed to initialize BlocksWorldValidator for binary encoding: {e}")
            return
    elif args.encoding_type == "sas":
        try:
            # SAS encoding does not require a predicate manifest file.
            validator = BlocksWorldValidator(num_blocks, args.encoding_type)
            print(f"Validator initialized with state_size={validator.state_size}.")
        except Exception as e:
            print(f"ERROR: Failed to initialize BlocksWorldValidator for SAS encoding: {e}")
            return
    else:
        print(f"ERROR: Unknown encoding type '{args.encoding_type}'")
        return
    print(f"Validator initialized with state_size={validator.state_size}.")

    # Initialize Model
    try:
        wrapped_model: PlannableModel = get_plannable_model(model_type, model_path, num_blocks, DEVICE)
    except ValueError as e:
        print(f"ERROR: Could not initialize model: {e}")
        return  # or exit
    except NotImplementedError as e:
        print(f"ERROR: Model type not implemented: {e}")
        return  # or exit

    wrapped_model.load_model()

    # Verify state_dim consistency
    if wrapped_model.state_dim != validator.state_size:
        print("CRITICAL ERROR: State dimension mismatch!")
        print(f"  Model ({wrapped_model.model_name}) expects/produces state_dim = {wrapped_model.state_dim}")
        print(f"  Validator (from manifest/config) expects state_size = {validator.state_size}")
        print("  Ensure the model was trained with data generated using this manifest or an equivalent encoding.")
        return

    print(f"Benchmarking model: {wrapped_model.model_name}")

    all_validation_results: List[ValidationResult] = []
    all_expert_plan_lengths: List[int] = []
    all_timeseries_metrics: List[Dict[str, float]] = []

    # Max plan length for generation (can be different from TTM's fixed prediction length)
    # For LSTM, this limits the generation loop. For TTM, its output is fixed length.
    # The validator will check if goal is met within the *generated* plan.
    max_generation_steps = args.max_plan_length

    for i, basename in enumerate(problem_basenames):
        print(f"  Processing problem {i + 1}/{len(problem_basenames)}: {basename} ...", end="", flush=True)

        traj_bin_path = data_root_dir / "trajectories_bin" / f"{basename}.traj.{args.encoding_type}.npy"
        goal_bin_path = data_root_dir / "trajectories_bin" / f"{basename}.goal.{args.encoding_type}.npy"

        if not traj_bin_path.exists() or not goal_bin_path.exists():
            print(" skipped (data files missing).")
            continue

        try:
            expert_trajectory_np = np.load(traj_bin_path)  # (L_expert, F)
            goal_state_np = np.load(goal_bin_path)  # (F,)
        except Exception as e:
            print(f" skipped (error loading npy: {e}).")
            continue

        if expert_trajectory_np.ndim != 2 or expert_trajectory_np.shape[0] == 0:
            print(f" skipped (expert trajectory malformed: shape {expert_trajectory_np.shape}).")
            continue

        initial_state_np = expert_trajectory_np[0]  # S0 (F,)
        all_expert_plan_lengths.append(expert_trajectory_np.shape[0])

        # Predict plan
        predicted_plan_list_of_lists = wrapped_model.predict_sequence(initial_state_np, goal_state_np, max_generation_steps)

        # Validate
        validation_res = validator.validate_sequence(predicted_plan_list_of_lists, goal_state_np.tolist())  # type: ignore
        all_validation_results.append(validation_res)

        # Compute and store time-series metrics
        ts_metrics = compute_timeseries_metrics(predicted_plan_list_of_lists, expert_trajectory_np, args.encoding_type)
        all_timeseries_metrics.append(ts_metrics)

        status = "VALID" if validation_res.is_valid else "INVALID"
        goal_reached = "GOAL" if validation_res.metrics.get("goal_achievement") == 1.0 else "NO_GOAL"
        print(
            f" done. Status: {status}, {goal_reached}. Len: {validation_res.predicted_plan_length}. DTW: {ts_metrics['dtw_distance']:.2f}"
        )

    # Aggregate and save results
    print("\nAggregating results...")
    aggregated_metrics = compute_aggregated_metrics(all_validation_results, all_expert_plan_lengths, all_timeseries_metrics)

    print("\n** Aggregated Benchmark Metrics **")
    for key, value in aggregated_metrics.items():
        if isinstance(value, dict) or isinstance(value, Counter):  # For violation_code_counts
            print(f"  {key}:")
            for k_inner, v_inner in value.items():
                print(f"    {k_inner}: {v_inner}")
        elif isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    results_filename = f"benchmark_results_{model_type}_{Path(model_path).stem}_N{num_blocks}.json"
    results_filepath = output_dir / results_filename
    with open(results_filepath, "w") as f:
        # Convert Counter to dict for JSON
        if "violation_code_counts" in aggregated_metrics and isinstance(
            aggregated_metrics["violation_code_counts"], Counter
        ):
            aggregated_metrics["violation_code_counts"] = dict(aggregated_metrics["violation_code_counts"])
        json.dump(aggregated_metrics, f, indent=4)
    print(f"\nBenchmark results saved to: {results_filepath}")

    # Optionally save detailed per-problem results
    if args.save_detailed_results:
        detailed_results_list = []
        for i, res in enumerate(all_validation_results):
            # Get the corresponding time-series metrics
            ts_metrics_for_problem = all_timeseries_metrics[i] if i < len(all_timeseries_metrics) else {}

            detailed_entry = {
                "problem_basename": problem_basenames[i] if i < len(problem_basenames) else "N/A",
                "is_valid": res.is_valid,
                "predicted_plan_length": res.predicted_plan_length,
                "expert_plan_length": all_expert_plan_lengths[i] if i < len(all_expert_plan_lengths) else -1,
                "goal_achievement": res.metrics.get("goal_achievement", 0.0),
                "goal_jaccard_score": res.goal_jaccard_score,
                "goal_f1_score": res.goal_f1_score,
                "timeseries_metrics": ts_metrics_for_problem,
                "violations": [{"code": v.code, "message": v.message, "details": v.details} for v in res.violations],
            }
            detailed_results_list.append(detailed_entry)

        detailed_filename = f"detailed_benchmark_results_{model_type}_{Path(model_path).stem}_N{num_blocks}.json"
        detailed_filepath = output_dir / detailed_filename
        with open(detailed_filepath, "w") as f_detailed:
            json.dump(detailed_results_list, f_detailed, indent=2)
        print(f"Detailed per-problem results saved to: {detailed_filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PaTS Model Benchmarking Script")
    parser.add_argument(
        "--dataset_dir", type=str, required=True, help="Directory of the PaTS dataset (e.g., 'data/blocks_4')"
    )
    parser.add_argument("--num_blocks", type=int, required=True, help="Number of blocks for the problems to benchmark.")
    parser.add_argument(
        "--model_type", type=str, required=True, choices=["ttm", "lstm"], help="Type of model to benchmark."
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the trained model directory (TTM) or .pth file (LSTM)."
    )
    parser.add_argument(
        "--encoding_type",
        type=str,
        default="bin",
        choices=["bin", "sas"],
        help="The encoding type of the dataset to benchmark against.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./benchmark_outputs", help="Directory to save benchmark results."
    )
    parser.add_argument("--max_plan_length", type=int, default=50, help="Maximum plan length for model generation.")
    parser.add_argument("--save_detailed_results", action="store_true", help="Save detailed results for each problem.")

    cli_args = parser.parse_args()
    run_benchmark(cli_args)

```

- `PaTS/scripts/BlocksWorldValidator.py`
```py
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class Violation:
    code: str  # e.g., "PHYS_BLOCK_FLOATING", "TRANS_ILLEGAL_CHANGES"
    message: str
    details: Optional[Dict[str, Any]] = None  # e.g., {"block": "A"}


@dataclass
class ValidationResult:
    is_valid: bool
    violations: List[Violation]  # Changed from List[str]
    metrics: Dict[str, float]
    # New metrics
    goal_jaccard_score: float = 0.0
    goal_f1_score: float = 0.0
    # Add a field to store the predicted plan itself for easier aggregation later
    predicted_plan_length: int = 0


class BlocksWorldValidator:
    def __init__(self, num_blocks: int, encoding_type: str, predicate_manifest_file: Optional[str | Path] = None):
        """
        Initialize validator for a specific number of blocks and encoding type.

        :param num_blocks: The number of blocks in the domain.
        :param encoding_type: The encoding type, either 'bin' or 'sas'.
        :param predicate_manifest_file: Path to the predicate manifest. Required for 'bin' encoding.
        """
        self.num_blocks = num_blocks
        self.encoding_type = encoding_type
        self.block_names: List[str] = [f"b{i + 1}" for i in range(self.num_blocks)]

        if self.encoding_type == "bin":
            if predicate_manifest_file is None:
                raise ValueError("predicate_manifest_file is required for 'bin' encoding.")
            self.predicate_manifest_file = Path(predicate_manifest_file)
            # These will be populated by _setup_feature_indices_binary
            self.predicate_list: List[str] = []
            self.on_table_indices: Dict[str, int] = {}
            self.on_block_indices: Dict[Tuple[str, str], int] = {}
            self.clear_indices: Dict[str, int] = {}
            self.held_indices: Dict[str, int] = {}
            self.arm_empty_index: Optional[int] = None
            self._on_table_idx_list: List[int] = []
            self._on_block_idx_list: List[int] = []
            self._clear_idx_list: List[int] = []
            self._held_idx_list: List[int] = []
            self._setup_feature_indices_binary()
            self.state_size = len(self.predicate_list)
        elif self.encoding_type == "sas":
            self.state_size = self.num_blocks
        else:
            raise ValueError(f"Unsupported encoding_type: {encoding_type}")

    def _setup_feature_indices_binary(self):
        """Calculate indices for binary predicate encoding."""
        if not self.predicate_manifest_file.is_file():
            raise FileNotFoundError(f"Predicate manifest file not found: {self.predicate_manifest_file}")

        with open(self.predicate_manifest_file, "r") as f:
            self.predicate_list = [line.strip() for line in f if line.strip()]

        self.state_size = len(self.predicate_list)
        if self.state_size == 0:
            raise ValueError(f"Predicate manifest file {self.predicate_manifest_file} is empty or invalid.")

        # Determine block names based on num_blocks, assuming "bX" convention from parse_and_encode.py
        self.block_names = [f"b{i + 1}" for i in range(self.num_blocks)]

        for idx, pred_string_from_manifest in enumerate(self.predicate_list):
            pred_string = pred_string_from_manifest.lower()  # Already normalized by parser

            # On table: (on-table bX)
            m_on_table = re.fullmatch(r"\(on-table (b\d+)\)", pred_string)
            if m_on_table:
                block_name = m_on_table.group(1)
                if block_name in self.block_names:
                    self.on_table_indices[block_name] = idx
                continue

            # On block: (on bX bY)
            m_on_block = re.fullmatch(r"\(on (b\d+) (b\d+)\)", pred_string)
            if m_on_block:
                b1 = m_on_block.group(1)
                b2 = m_on_block.group(2)
                if b1 in self.block_names and b2 in self.block_names and b1 != b2:
                    self.on_block_indices[(b1, b2)] = idx
                continue

            # Clear: (clear bX)
            m_clear = re.fullmatch(r"\(clear (b\d+)\)", pred_string)
            if m_clear:
                block_name = m_clear.group(1)
                if block_name in self.block_names:
                    self.clear_indices[block_name] = idx
                continue

            # Arm empty: (arm-empty)
            if pred_string == "(arm-empty)":
                self.arm_empty_index = idx
                continue

            # Holding: (holding bX)
            m_holding = re.fullmatch(r"\(holding (b\d+)\)", pred_string)
            if m_holding:
                block_name = m_holding.group(1)
                if block_name in self.block_names:
                    self.held_indices[block_name] = idx
                continue

        # Populate the index lists for tensor operations
        self._on_table_idx_list = list(self.on_table_indices.values())
        self._on_block_idx_list = list(self.on_block_indices.values())
        self._clear_idx_list = list(self.clear_indices.values())
        self._held_idx_list = list(self.held_indices.values())

        # Sanity check: ensure arm_empty_index was found
        if self.arm_empty_index is None:
            raise ValueError("(arm-empty) predicate not found in manifest. This is required.")

    def calculate_constraint_violation_loss(self, state_logits: torch.Tensor) -> torch.Tensor:
        """Dispatch to the correct loss function based on encoding."""
        if self.encoding_type == "bin":
            return self._calculate_constraint_violation_loss_binary(state_logits)
        elif self.encoding_type == "sas":
            # NOTE: A differentiable loss for SAS+ is non-trivial as the values are discrete and interdependent.
            # A proper implementation might use a cross-entropy loss over possible locations for each block.
            # For now, we return zero loss, effectively disabling this feature for SAS+.
            return torch.tensor(0.0, device=state_logits.device)
        return torch.tensor(0.0, device=state_logits.device)

    def _calculate_constraint_violation_loss_binary(self, state_logits: torch.Tensor) -> torch.Tensor:
        """
        Calculates a differentiable loss term based on physical constraint violations.
        Operates on a batch of state logits from the model.

        :param state_logits: A tensor of shape (N, F) where N is the number of states in the batch and F is the number of features.
        :return: A scalar tensor representing the mean violation loss per state.
        """
        if state_logits.shape[0] == 0:
            return torch.tensor(0.0, device=state_logits.device)

        # Use probabilities for a smoother loss landscape
        state_probs = torch.sigmoid(state_logits)
        total_loss = torch.tensor(0.0, device=state_probs.device)
        num_states = state_probs.shape[0]

        # 1. PHYS_BLOCK_MULTI_POS: Each block must be in exactly one position.
        # (on table, on another block, or held).
        for block_name in self.block_names:
            pos_indices = []
            if block_name in self.on_table_indices:
                pos_indices.append(self.on_table_indices[block_name])
            if block_name in self.held_indices:
                pos_indices.append(self.held_indices[block_name])
            for b1, b2 in self.on_block_indices.keys():
                if b1 == block_name:
                    pos_indices.append(self.on_block_indices[(b1, b2)])

            # Sum the probabilities of the block being in any position
            pos_probs_sum = state_probs[:, pos_indices].sum(dim=1)
            # The sum should be 1.0. Penalize deviation from 1.0.
            total_loss += F.mse_loss(pos_probs_sum, torch.ones_like(pos_probs_sum))

        # 2. PHYS_CLEAR_CONFLICT: A block marked 'clear' cannot have another block on it.
        for block_name in self.block_names:
            clear_prob = state_probs[:, self.clear_indices[block_name]]

            # Sum probabilities of any other block being on top of this one
            on_top_indices = [
                self.on_block_indices[(other_block, block_name)]
                for other_block in self.block_names
                if other_block != block_name
            ]
            if on_top_indices:
                on_top_prob_sum = state_probs[:, on_top_indices].sum(dim=1)
                # Violation if both clear_prob and on_top_prob are high.
                # Their product should be 0.
                violation_prob = clear_prob * on_top_prob_sum
                total_loss += violation_prob.mean()

        # 3. PHYS_ARM_EMPTY_HOLDING_CONFLICT & PHYS_ARM_NOT_EMPTY_NOT_HOLDING_CONFLICT
        # Arm is empty IFF it is not holding any block.
        if self.arm_empty_index is not None and self._held_idx_list:
            arm_empty_prob = state_probs[:, self.arm_empty_index]
            is_holding_prob = state_probs[:, self._held_idx_list].sum(dim=1)

            # The sum of arm_empty_prob and is_holding_prob should be 1.0
            # e.g., if arm_empty is 0.9, is_holding should be 0.1.
            # This elegantly captures both conflict types.
            total_loss += F.mse_loss(arm_empty_prob + is_holding_prob, torch.ones_like(arm_empty_prob))

        return total_loss / num_states if num_states > 0 else torch.tensor(0.0)

    def _check_physical_constraints(self, state: List[int] | np.ndarray) -> Tuple[bool, List[Violation]]:
        """Dispatch to the correct physical constraint checker based on encoding."""
        if self.encoding_type == "bin":
            return self._check_physical_constraints_binary(state)
        elif self.encoding_type == "sas":
            return self._check_physical_constraints_sas(state)
        return False, [Violation("INTERNAL_ERROR", "Unknown encoding type in validator.")]

    def _check_physical_constraints_binary(self, state: List[int] | np.ndarray) -> Tuple[bool, List[Violation]]:
        """Check if state satisfies basic physical constraints"""
        violations: List[Violation] = []
        state_arr = np.array(state) if not isinstance(state, np.ndarray) else state

        if len(state_arr) != self.state_size:
            violations.append(
                Violation("STATE_INVALID_SIZE", f"Invalid state size: expected {self.state_size}, got {len(state_arr)}")
            )
            return False, violations

        # Check each block
        for block in self.block_names:
            positions = 0
            # On table
            if block in self.on_table_indices and state_arr[self.on_table_indices[block]] == 1:
                positions += 1
            # On another block
            for other_block in self.block_names:
                if other_block != block and (block, other_block) in self.on_block_indices:
                    if state_arr[self.on_block_indices[(block, other_block)]] == 1:
                        positions += 1
            # Held
            if block in self.held_indices and state_arr[self.held_indices[block]] == 1:
                positions += 1

            if positions == 0:
                violations.append(
                    Violation(
                        "PHYS_BLOCK_FLOATING",
                        f"Block {block} is floating (not on any surface or held)",
                        {"block": block},
                    )
                )
            elif positions > 1:
                violations.append(
                    Violation(
                        "PHYS_BLOCK_MULTI_POS",
                        f"Block {block} is in multiple positions simultaneously",
                        {"block": block},
                    )
                )

            # Check clear status consistency
            if block in self.clear_indices and state_arr[self.clear_indices[block]] == 1:
                for other_on_top in self.block_names:  # Check if any other_on_top is on 'block'
                    if other_on_top != block and (other_on_top, block) in self.on_block_indices:
                        if state_arr[self.on_block_indices[(other_on_top, block)]] == 1:
                            violations.append(
                                Violation(
                                    "PHYS_CLEAR_CONFLICT",
                                    f"Block {block} marked as clear but has {other_on_top} on top",
                                    {"block": block, "block_on_top": other_on_top},
                                )
                            )

        # Check arm-empty and holding consistency
        num_held_blocks = 0
        held_block_names = []
        for block_h in self.block_names:
            if block_h in self.held_indices and state_arr[self.held_indices[block_h]] == 1:
                num_held_blocks += 1
                held_block_names.append(block_h)

        if self.arm_empty_index is None:  # Should have been caught by _setup_feature_indices
            violations.append(Violation("INTERNAL_ERROR", "arm_empty_index not configured"))
            return False, violations  # Cannot proceed with this check

        is_arm_empty_predicate_true = state_arr[self.arm_empty_index] == 1

        if num_held_blocks > 1:
            violations.append(
                Violation(
                    "PHYS_ARM_MULTI_HOLD",
                    f"Arm is holding multiple blocks: {', '.join(held_block_names)}",
                    {"held_blocks": held_block_names},
                )
            )

        if is_arm_empty_predicate_true:
            if num_held_blocks > 0:
                violations.append(
                    Violation(
                        "PHYS_ARM_EMPTY_HOLDING_CONFLICT",
                        f"Arm-empty predicate is true, but arm is holding block(s): {', '.join(held_block_names)}",
                        {"held_blocks": held_block_names},
                    )
                )
        else:  # Arm-empty predicate is false (i.e., arm should be holding something)
            if num_held_blocks == 0:
                violations.append(
                    Violation(
                        "PHYS_ARM_NOT_EMPTY_NOT_HOLDING_CONFLICT",
                        "Arm-empty predicate is false, but arm is not holding any block.",
                    )
                )

        return len(violations) == 0, violations

    def _check_physical_constraints_sas(self, state: List[int] | np.ndarray) -> Tuple[bool, List[Violation]]:
        """Checks physical constraints for a SAS+ encoded state vector."""
        violations: List[Violation] = []
        state_arr = np.array(state, dtype=int) if not isinstance(state, np.ndarray) else state.astype(int)

        if len(state_arr) != self.state_size:
            violations.append(
                Violation("STATE_INVALID_SIZE", f"Invalid state size: expected {self.state_size}, got {len(state_arr)}")
            )
            return False, violations

        # 1. Check for multiple held blocks
        held_indices = np.where(state_arr == -1)[0]
        if len(held_indices) > 1:
            held_blocks = [self.block_names[i] for i in held_indices]
            violations.append(
                Violation(
                    "PHYS_ARM_MULTI_HOLD", f"Arm is holding multiple blocks: {held_blocks}", {"held_blocks": held_blocks}
                )
            )

        # 2. Check for invalid positions and build a "support" dictionary
        support_map = {}  # key=block_idx, value=supported_by_idx (0 for table, -1 for arm)
        for i, pos in enumerate(state_arr):
            if pos > self.num_blocks or pos < -1:
                violations.append(
                    Violation("PHYS_INVALID_POS", f"Block {self.block_names[i]} has invalid position value {pos}")
                )
            if pos == i + 1:
                violations.append(Violation("PHYS_SELF_SUPPORT", f"Block {self.block_names[i]} cannot be on itself."))
            if pos > 0:  # on another block
                support_map[i] = pos - 1  # store 0-based index of supporter
            elif pos == 0:  # on table
                support_map[i] = "table"
            elif pos == -1:  # held
                support_map[i] = "arm"

        # 3. Check for support cycles (e.g., b1 on b2, b2 on b1)
        for block_idx in range(self.num_blocks):
            path = [block_idx]
            curr = block_idx
            while curr in support_map and support_map[curr] not in ["table", "arm"]:
                curr = support_map[curr]
                if curr in path:
                    cycle_names = [self.block_names[p] for p in path] + [self.block_names[curr]]
                    violations.append(
                        Violation("PHYS_SUPPORT_CYCLE", f"Support cycle detected: {' -> '.join(cycle_names)}")
                    )
                    break  # Found a cycle, break inner while
                path.append(curr)

        # 4. Check for multiple blocks on the same support
        on_top_of = {}  # key=supporter_idx, value=list of blocks on top
        for i, pos in enumerate(state_arr):
            if pos > 0:  # on another block
                supporter_idx = pos - 1
                if supporter_idx not in on_top_of:
                    on_top_of[supporter_idx] = []
                on_top_of[supporter_idx].append(i)

        for supporter_idx, blocks_on_top in on_top_of.items():
            if len(blocks_on_top) > 1:
                supporter_name = self.block_names[supporter_idx]
                block_names_on_top = [self.block_names[b] for b in blocks_on_top]
                violations.append(
                    Violation("PHYS_MULTI_ON_TOP", f"Multiple blocks ({block_names_on_top}) on top of {supporter_name}")
                )

        return len(violations) == 0, violations

    def _check_legal_transition(self, state1: np.ndarray, state2: np.ndarray) -> Tuple[bool, List[Violation]]:
        """Dispatch to the correct transition checker based on encoding."""
        if self.encoding_type == "bin":
            return self._check_legal_transition_binary(state1, state2)
        elif self.encoding_type == "sas":
            return self._check_legal_transition_sas(state1, state2)
        return False, [Violation("INTERNAL_ERROR", "Unknown encoding type in validator.")]

    def _check_legal_transition_binary(self, state1: np.ndarray, state2: np.ndarray) -> Tuple[bool, List[Violation]]:
        """Check if transition between states is legal according to blocks world rules.
        Assumes state1 and state2 are individually physically valid."""
        violations: List[Violation] = []
        # Ensure states are numpy arrays for efficient comparison
        s1 = np.array(state1) if not isinstance(state1, np.ndarray) else state1
        s2 = np.array(state2) if not isinstance(state2, np.ndarray) else state2

        differences = np.sum(s1 != s2)

        if differences == 0:
            # This is not necessarily an error if the state is the goal state.
            # The validate_sequence method will handle this context.
            violations.append(Violation("TRANS_NO_CHANGE", "No change between states", {"diff_count": 0}))
        # A single action (pickup, putdown) changes 4 features.
        # A single action (stack, unstack) changes 5 features.
        elif differences < 4 or differences > 5:
            violations.append(
                Violation(
                    "TRANS_ILLEGAL_CHANGES",
                    f"Illegal number of changes ({differences} bits changed). Expected 4 or 5 for a single action.",
                    {"diff_count": int(differences)},
                )
            )

        return len(violations) == 0, violations

    def _check_legal_transition_sas(self, state1: np.ndarray, state2: np.ndarray) -> Tuple[bool, List[Violation]]:
        """Checks if transition between SAS+ states is legal."""
        violations: List[Violation] = []
        s1 = np.array(state1, dtype=int)
        s2 = np.array(state2, dtype=int)

        diff_indices = np.where(s1 != s2)[0]
        differences = len(diff_indices)

        if differences == 0:
            violations.append(Violation("TRANS_NO_CHANGE", "No change between states", {"diff_count": 0}))
        # A valid action (pickup, putdown, stack, unstack) changes the position of exactly ONE block.
        elif differences != 1:
            changed_blocks = [self.block_names[i] for i in diff_indices]
            violations.append(
                Violation(
                    "TRANS_ILLEGAL_CHANGES",
                    f"Illegal number of changes ({differences}). Expected 1 block to change position. Changed: {changed_blocks}",
                    {"diff_count": int(differences), "changed_blocks": changed_blocks},
                )
            )
        # Further checks could be added here to validate the specific change (e.g., was the moved block clear in s1?)
        # but for now, checking the number of changes is a strong heuristic.

        return len(violations) == 0, violations

    def validate_sequence(
        self, states: List[List[int] | np.ndarray], goal_state: List[int] | np.ndarray
    ) -> ValidationResult:
        """Validate a complete sequence of states leading to a goal"""
        all_violations_obj: List[Violation] = []
        metrics: Dict[str, float] = {}

        # Convert all states to numpy arrays for consistency
        np_states = [np.array(s) for s in states]
        np_goal_state = np.array(goal_state)

        if not np_states:
            all_violations_obj.append(Violation("SEQ_EMPTY", "Predicted sequence is empty."))
            metrics["sequence_length"] = 0
            metrics["goal_achievement"] = 0.0
            metrics["avg_changes_per_step"] = 0.0
            return ValidationResult(
                is_valid=False,
                violations=all_violations_obj,
                metrics=metrics,
                goal_jaccard_score=0.0,
                goal_f1_score=0.0,
                predicted_plan_length=0,
            )

        # 1. Validate all states individually for physical constraints
        physically_valid_states_count = 0
        for i, state_vector in enumerate(np_states):
            is_physically_valid, physical_violations = self._check_physical_constraints(state_vector)
            if not is_physically_valid:
                for v in physical_violations:
                    all_violations_obj.append(Violation(v.code, f"State {i} invalid: {v.message}", v.details))
            else:
                physically_valid_states_count += 1

        metrics["percent_physically_valid_states"] = (
            (physically_valid_states_count / len(np_states)) * 100 if np_states else 0.0
        )

        # If any state is physically invalid, the sequence is fundamentally flawed for some metrics,
        # but we can still report others.
        # Let's continue to check transitions if desired, or bail early.
        # For now, let's assume we continue to gather all possible violations.

        # 2. Check transitions.
        valid_transitions_count = 0
        if len(np_states) > 1:
            for i in range(len(np_states) - 1):
                # Pass np_states[i] and np_states[i+1]
                valid_transition, transition_violations_list = self._check_legal_transition(np_states[i], np_states[i + 1])

                is_acceptable_no_change = False
                if not valid_transition:
                    # Check if the only violation is "No change between states" AND the state is the goal state
                    if any(v.code == "TRANS_NO_CHANGE" for v in transition_violations_list):
                        if np.array_equal(np_states[i], np_goal_state):
                            if (
                                len(transition_violations_list) == 1
                                and transition_violations_list[0].code == "TRANS_NO_CHANGE"
                            ):
                                is_acceptable_no_change = True
                                # This "no change at goal" is fine, counts as a valid step in a sense
                                valid_transitions_count += 1

                    if not is_acceptable_no_change:
                        for v_trans in transition_violations_list:
                            all_violations_obj.append(
                                Violation(v_trans.code, f"Transition {i}->{i + 1}: {v_trans.message}", v_trans.details)
                            )
                else:
                    valid_transitions_count += 1
            metrics["percent_valid_transitions"] = (valid_transitions_count / (len(np_states) - 1)) * 100
        else:  # single state sequence
            metrics["percent_valid_transitions"] = 100.0  # Or 0.0, or N/A. If S0 is goal, it's valid.

        # 3. Check if the final state of the sequence matches the goal state
        final_state_is_goal = np.array_equal(np_states[-1], np_goal_state)
        if not final_state_is_goal:
            all_violations_obj.append(Violation("SEQ_GOAL_MISMATCH", "Final state does not match goal state."))

        metrics["goal_achievement"] = float(final_state_is_goal)

        # Calculate Jaccard and F1 for goal match
        final_pred_state = np_states[-1]
        pred_true_indices = set(np.where(final_pred_state == 1)[0])
        goal_true_indices = set(np.where(np_goal_state == 1)[0])

        intersection_len = len(pred_true_indices.intersection(goal_true_indices))
        union_len = len(pred_true_indices.union(goal_true_indices))

        jaccard = (
            intersection_len / union_len
            if union_len > 0
            else (1.0 if not pred_true_indices and not goal_true_indices else 0.0)
        )

        precision = (
            intersection_len / len(pred_true_indices)
            if len(pred_true_indices) > 0
            else (1.0 if intersection_len == 0 and len(goal_true_indices) == 0 else 0.0)
        )
        recall = (
            intersection_len / len(goal_true_indices)
            if len(goal_true_indices) > 0
            else (1.0 if intersection_len == 0 and len(pred_true_indices) == 0 else 0.0)
        )

        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else (1.0 if precision == 1.0 and recall == 1.0 else 0.0)
        )

        # Populate metrics
        metrics["sequence_length"] = len(np_states)

        if len(np_states) > 1:
            sum_changes_for_avg = 0
            for i_tc in range(len(np_states) - 1):
                diff_count = np.sum(np_states[i_tc] != np_states[i_tc + 1])
                sum_changes_for_avg += diff_count
            metrics["avg_changes_per_step"] = sum_changes_for_avg / (len(np_states) - 1)
        else:
            metrics["avg_changes_per_step"] = 0.0

        # Overall validity: no violations at all.
        # The definition of "is_valid" might need refinement.
        # For now, let's say it's valid if no *critical* violations.
        # Or, simply, if all_violations_obj is empty.
        # The original logic was: if all_violations is empty. Let's stick to that for now.
        # "TRANS_NO_CHANGE" at goal is acceptable and shouldn't make it invalid.
        # We need to filter out acceptable "TRANS_NO_CHANGE" violations before checking if all_violations_obj is empty.

        final_violations_for_is_valid_check = []
        for v_obj in all_violations_obj:
            if v_obj.code == "TRANS_NO_CHANGE":
                # Check if this "no change" occurred when the state was already the goal
                # This requires knowing which state index this transition violation refers to.
                # The message "Transition {i}->{i+1}" helps.
                match_trans_idx = re.search(r"Transition (\d+)->(\d+)", v_obj.message)
                if match_trans_idx:
                    state_idx_before_no_change = int(match_trans_idx.group(1))
                    if np.array_equal(np_states[state_idx_before_no_change], np_goal_state):
                        continue  # This "no change" is acceptable, don't count for overall invalidity
            final_violations_for_is_valid_check.append(v_obj)

        return ValidationResult(
            is_valid=len(final_violations_for_is_valid_check) == 0,
            violations=all_violations_obj,  # Return all, even if some are acceptable for is_valid
            metrics=metrics,
            goal_jaccard_score=jaccard,
            goal_f1_score=f1,
            predicted_plan_length=len(np_states),
        )

```

- `PaTS/scripts/pats_dataset.py`
```py
from pathlib import Path
from typing import Any, Dict

import numpy as np
from torch.utils.data import Dataset


class PaTSDataset(Dataset):
    """
    A PyTorch Dataset for loading Planning as Time-Series (PaTS) data.
    It loads pre-encoded binary trajectories and goal states from .npy files.
    """

    def __init__(self, dataset_dir: str | Path, split_file_name: str, encoding_type: str = "bin"):
        """
        Initializes the PaTSDataset.

        :param dataset_dir: The root directory for a specific number of blocks (e.g., 'data/blocks_4/').
        :param split_file_name: The name of the file containing problem basenames for this split (e.g., 'train_files.txt').
        :param encoding_type: The encoding of the data to load ('bin' or 'sas').
        """
        self.dataset_dir = Path(dataset_dir)
        self.split_file_path = self.dataset_dir / split_file_name
        self.encoding_type = encoding_type

        self.trajectories_bin_dir = self.dataset_dir / "trajectories_bin"
        if not self.trajectories_bin_dir.is_dir():
            raise FileNotFoundError(f"Trajectories binary directory not found: {self.trajectories_bin_dir}")

        self.basenames = self._load_basenames()
        if not self.basenames:
            raise ValueError(
                f"No basenames loaded from {self.split_file_path}. "
                "Ensure the file exists, is not empty, and is in the correct directory."
            )

        self.state_dim = self._infer_state_dim()
        if self.state_dim <= 0:
            raise ValueError(f"Inferred state_dim is {self.state_dim}, which is invalid.")

    def _load_basenames(self) -> list[str]:
        if not self.split_file_path.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_file_path}")
        with open(self.split_file_path, "r") as f:
            basenames = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        return basenames

    def _infer_state_dim(self) -> int:
        # Try to load the first trajectory to get state_dim
        # This assumes all trajectories in the dataset share the same state dimension.
        if not self.basenames:
            # This should have been caught by the check in __init__ after _load_basenames
            raise RuntimeError("Cannot infer state_dim: no basenames available.")

        first_basename = self.basenames[0]
        traj_path = self.trajectories_bin_dir / f"{first_basename}.traj.{self.encoding_type}.npy"
        if not traj_path.exists():
            raise FileNotFoundError(
                f"Trajectory file for state_dim inference not found: {traj_path}. Checked for basename: {first_basename}"
            )

        try:
            trajectory_np = np.load(traj_path)
            if trajectory_np.ndim == 2 and trajectory_np.shape[0] > 0:
                return trajectory_np.shape[1]
            else:
                raise ValueError(
                    f"Trajectory file {traj_path} is empty or malformed for state_dim inference. "
                    f"Shape: {trajectory_np.shape}"
                )
        except Exception as e:
            raise IOError(f"Error loading trajectory {traj_path} for state_dim inference: {e}") from e

    def __len__(self) -> int:
        return len(self.basenames)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        basename = self.basenames[idx]

        traj_path = self.trajectories_bin_dir / f"{basename}.traj.{self.encoding_type}.npy"
        goal_path = self.trajectories_bin_dir / f"{basename}.goal.{self.encoding_type}.npy"

        if not traj_path.exists():
            raise FileNotFoundError(f"Trajectory file not found for basename {basename}: {traj_path}")
        if not goal_path.exists():
            raise FileNotFoundError(f"Goal file not found for basename {basename}: {goal_path}")

        try:
            # Load trajectory and goal based on encoding type
            if self.encoding_type == "sas":
                # For SAS+, we need integer types for embedding layers and loss functions
                expert_trajectory_np = np.load(traj_path).astype(np.int64)
                goal_state_np = np.load(goal_path).astype(np.int64)
            else:  # binary
                expert_trajectory_np = np.load(traj_path).astype(np.float32)
                goal_state_np = np.load(goal_path).astype(np.float32)

        except Exception as e:
            raise IOError(f"Failed to load .npy files for basename {basename} (idx {idx}): {e}") from e

        # Validate shapes and consistency
        if expert_trajectory_np.ndim != 2 or expert_trajectory_np.shape[0] == 0:
            raise ValueError(
                f"Expert trajectory for {basename} is malformed or empty. "
                f"Shape: {expert_trajectory_np.shape}, Expected: (L > 0, F > 0)"
            )
        if goal_state_np.ndim != 1:
            raise ValueError(f"Goal state for {basename} is malformed. Shape: {goal_state_np.shape}, Expected: (F > 0,)")
        if expert_trajectory_np.shape[1] != self.state_dim:
            raise ValueError(
                f"Feature dimension mismatch in trajectory for {basename}. "
                f"Expected {self.state_dim} (from first item), got {expert_trajectory_np.shape[1]}"
            )
        if goal_state_np.shape[0] != self.state_dim:
            raise ValueError(
                f"Feature dimension mismatch in goal for {basename}. "
                f"Expected {self.state_dim} (from first item), got {goal_state_np.shape[0]}"
            )

        initial_state_np = expert_trajectory_np[0, :]  # (F,)

        return {
            "initial_state": initial_state_np,
            "goal_state": goal_state_np,
            "expert_trajectory": expert_trajectory_np,
            "id": basename,
        }

```

- `PaTS/scripts/train_model.py`
```py
import argparse
import sys
import warnings
from datetime import datetime
from functools import partial
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers.trainer_utils import set_seed as ttm_set_seed

from scripts.BlocksWorldValidator import BlocksWorldValidator
from scripts.models.lstm import PaTS_LSTM, lstm_collate_fn
from scripts.models.ttm import BlocksWorldTTM, determine_ttm_model
from scripts.models.ttm import ModelConfig as TTMModelConfig
from scripts.models.ttm import setup_logging as ttm_setup_logging
from scripts.pats_dataset import PaTSDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))


def train_lstm_model_loop(model, train_loader, val_loader, validator, args, num_features, model_save_path):
    print("Starting LSTM training...")

    # Select Loss Function based on Encoding
    if args.encoding_type == "sas":
        # For SAS+, the model outputs logits for each possible location (class) for each block.
        # The target is the index of the correct location.
        criterion = nn.CrossEntropyLoss()
        print("Using CrossEntropyLoss for SAS+ encoding.")
    else:  # binary
        criterion = nn.BCEWithLogitsLoss(reduction="none")
        print("Using BCEWithLogitsLoss for binary encoding.")

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=10, factor=0.5)

    best_val_loss = float("inf")
    sas_clamp_warning_issued = False

    for epoch in range(args.epochs):
        model.train()
        epoch_train_loss, epoch_forecast_loss, epoch_mlm_loss, epoch_constraint_loss = 0.0, 0.0, 0.0, 0.0
        num_train_batches = 0

        for batch_data in train_loader:
            if batch_data is None:
                continue
            input_seqs = batch_data["input_sequences"].to(DEVICE)
            goal_states = batch_data["goal_states"].to(DEVICE)
            target_seqs = batch_data["target_sequences"].to(DEVICE)
            lengths = batch_data["lengths"]
            # MLM mask is only relevant for binary encoding
            mlm_predicate_mask = batch_data.get("mlm_predicate_mask", torch.tensor([])).to(DEVICE)

            # Safeguard for SAS+ encoding to prevent out-of-bounds embedding errors.
            if args.encoding_type == "sas":
                max_val = args.num_blocks
                # Check if any values are out of the expected [0, max_val] range.
                if not sas_clamp_warning_issued and (torch.any(input_seqs > max_val) or torch.any(goal_states > max_val)):
                    warnings.warn(
                        f"SAS+ input data contains values > num_blocks ({max_val}). "
                        f"Clamping values to prevent embedding layer errors. "
                        f"Please check your dataset for correctness."
                    )
                    sas_clamp_warning_issued = True
                input_seqs.clamp_(min=0, max=max_val)
                goal_states.clamp_(min=0, max=max_val)

            optimizer.zero_grad()
            # For SAS+, mlm_logits will be None
            forecasting_logits, mlm_logits, _ = model(input_seqs, goal_states, lengths)

            # Modified Loss Calculation
            if args.encoding_type == "sas":
                # For CrossEntropyLoss, logits should be (N, C) and targets (N)
                # N = total number of blocks to predict across batch, C = num_locations
                # Create a mask to select only the valid time steps based on sequence lengths
                mask = (
                    torch.arange(max(lengths), device=DEVICE)[None, :] < lengths.clone().detach().to(DEVICE)[:, None]
                )  # (B, S_max)

                # Reshape logits and targets and apply the mask
                # Logits: (B, S_max, N_blocks, N_locs) -> (num_active_steps, N_blocks, N_locs)
                active_logits = forecasting_logits[mask]
                # Targets: (B, S_max, N_blocks) -> (num_active_steps, N_blocks)
                active_targets = target_seqs[mask]

                if active_targets.numel() == 0:
                    continue  # Skip batch if no valid targets

                # Map SAS+ target values (e.g., 0 for table, 1..N for blocks) to class indices
                active_targets_indices = model._map_sas_to_indices(active_targets)

                # Final reshape for loss function
                # (num_active_steps * N_blocks, N_locs) and (num_active_steps * N_blocks)
                loss_forecasting = criterion(
                    active_logits.reshape(-1, model.num_locations), active_targets_indices.reshape(-1)
                )
            else:  # Binary encoding loss calculation
                forecasting_mask = torch.zeros_like(target_seqs, dtype=torch.bool).to(DEVICE)
                for i, length_val in enumerate(lengths):
                    if length_val > 0:
                        forecasting_mask[i, :length_val, :] = True

                num_forecast_elements = forecasting_mask.float().sum()
                if num_forecast_elements == 0:
                    continue

                loss_forecasting_unreduced = criterion(forecasting_logits, target_seqs)
                loss_forecasting = (loss_forecasting_unreduced * forecasting_mask.float()).sum() / num_forecast_elements

            # For SAS+, MLM and Constraint losses are not used. They will be zero.
            loss_mlm = torch.tensor(0.0).to(DEVICE)
            if args.encoding_type != "sas" and args.use_mlm_task and mlm_logits is not None:
                num_masked_elements = mlm_predicate_mask.sum()
                if num_masked_elements > 0:
                    loss_mlm_unreduced = criterion(mlm_logits, input_seqs)
                    loss_mlm = (loss_mlm_unreduced * mlm_predicate_mask).sum() / num_masked_elements

            loss_constraint = torch.tensor(0.0).to(DEVICE)
            if args.encoding_type != "sas" and args.use_constraint_loss and validator is not None:
                masked_logits = forecasting_logits[forecasting_mask[:, :, 0]]
                if masked_logits.ndim == 2 and masked_logits.shape[0] > 0:
                    loss_constraint = validator.calculate_constraint_violation_loss(masked_logits)

            # Total Loss
            total_loss = loss_forecasting + args.mlm_loss_weight * loss_mlm + args.constraint_loss_weight * loss_constraint
            total_loss.backward()
            if args.clip_grad_norm is not None and args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

            epoch_train_loss += total_loss.item()
            epoch_forecast_loss += loss_forecasting.item()
            epoch_mlm_loss += loss_mlm.item()
            epoch_constraint_loss += loss_constraint.item()
            num_train_batches += 1

        avg_train_loss = epoch_train_loss / num_train_batches if num_train_batches > 0 else float("inf")
        avg_forecast_loss = epoch_forecast_loss / num_train_batches if num_train_batches > 0 else float("inf")
        avg_mlm_loss = epoch_mlm_loss / num_train_batches if num_train_batches > 0 else float("inf")
        avg_constraint_loss = epoch_constraint_loss / num_train_batches if num_train_batches > 0 else float("inf")

        # Validation
        model.eval()
        epoch_val_loss, epoch_val_forecast_loss, epoch_val_mlm_loss, epoch_val_constraint_loss = 0.0, 0.0, 0.0, 0.0
        num_val_batches = 0
        with torch.no_grad():
            if val_loader is not None:
                for batch_data in val_loader:
                    if batch_data is None:
                        continue
                    input_seqs = batch_data["input_sequences"].to(DEVICE)
                    goal_states = batch_data["goal_states"].to(DEVICE)
                    target_seqs = batch_data["target_sequences"].to(DEVICE)
                    lengths = batch_data["lengths"]
                    mlm_predicate_mask = batch_data.get("mlm_predicate_mask", torch.tensor([])).to(DEVICE)

                    if args.encoding_type == "sas":
                        input_seqs.clamp_(min=0, max=args.num_blocks)
                        goal_states.clamp_(min=0, max=args.num_blocks)

                    forecasting_logits, mlm_logits, _ = model(input_seqs, goal_states, lengths)

                    if args.encoding_type == "sas":
                        mask = (
                            torch.arange(max(lengths), device=DEVICE)[None, :]
                            < lengths.clone().detach().to(DEVICE)[:, None]
                        )
                        active_logits = forecasting_logits[mask]
                        active_targets = target_seqs[mask]

                        if active_targets.numel() == 0:
                            continue

                        active_targets_indices = model._map_sas_to_indices(active_targets)
                        loss_forecasting = criterion(
                            active_logits.reshape(-1, model.num_locations), active_targets_indices.reshape(-1)
                        )
                    else:  # Binary
                        forecasting_mask = torch.zeros_like(target_seqs, dtype=torch.bool).to(DEVICE)
                        for i, length_val in enumerate(lengths):
                            if length_val > 0:
                                forecasting_mask[i, :length_val, :] = True
                        num_forecast_elements = forecasting_mask.float().sum()
                        if num_forecast_elements == 0:
                            continue
                        loss_forecasting_unreduced = criterion(forecasting_logits, target_seqs)
                        loss_forecasting = (
                            loss_forecasting_unreduced * forecasting_mask.float()
                        ).sum() / num_forecast_elements

                    loss_mlm = torch.tensor(0.0).to(DEVICE)
                    if args.encoding_type != "sas" and args.use_mlm_task and mlm_logits is not None:
                        num_masked_elements = mlm_predicate_mask.sum()
                        if num_masked_elements > 0:
                            loss_mlm_unreduced = criterion(mlm_logits, input_seqs)
                            loss_mlm = (loss_mlm_unreduced * mlm_predicate_mask).sum() / num_masked_elements

                    loss_constraint = torch.tensor(0.0).to(DEVICE)
                    if args.encoding_type != "sas" and args.use_constraint_loss and validator is not None:
                        masked_logits = forecasting_logits[forecasting_mask[:, :, 0]]
                        if masked_logits.ndim == 2 and masked_logits.shape[0] > 0:
                            loss_constraint = validator.calculate_constraint_violation_loss(masked_logits)

                    total_loss = (
                        loss_forecasting + args.mlm_loss_weight * loss_mlm + args.constraint_loss_weight * loss_constraint
                    )
                    epoch_val_loss += total_loss.item()
                    epoch_val_forecast_loss += loss_forecasting.item()
                    epoch_val_mlm_loss += loss_mlm.item()
                    epoch_val_constraint_loss += loss_constraint.item()
                    num_val_batches += 1

        avg_val_loss = epoch_val_loss / num_val_batches if num_val_batches > 0 else float("inf")
        avg_val_forecast_loss = epoch_val_forecast_loss / num_val_batches if num_val_batches > 0 else float("inf")
        avg_val_mlm_loss = epoch_val_mlm_loss / num_val_batches if num_val_batches > 0 else float("inf")
        avg_val_constraint_loss = epoch_val_constraint_loss / num_val_batches if num_val_batches > 0 else float("inf")
        scheduler.step(avg_val_loss)

        train_loss_str = f"Train Loss: {avg_train_loss:.4f} (F: {avg_forecast_loss:.4f}, M: {avg_mlm_loss:.4f}, C: {avg_constraint_loss:.4f})"
        val_loss_str = f"Val Loss: {avg_val_loss:.4f} (F: {avg_val_forecast_loss:.4f}, M: {avg_val_mlm_loss:.4f}, C: {avg_val_constraint_loss:.4f})"
        print(f"Epoch [{epoch + 1}/{args.epochs}] {train_loss_str}, {val_loss_str}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_val_loss,
                    "encoding_type": args.encoding_type,
                    "num_features": num_features,
                    "hidden_size": args.lstm_hidden_size,
                    "num_lstm_layers": args.lstm_num_layers,
                    "dropout_prob": args.lstm_dropout_prob,
                    "use_mlm_task": args.use_mlm_task,
                    "mlm_loss_weight": args.mlm_loss_weight,
                    "target_num_blocks": args.num_blocks,
                    "embedding_dim": args.lstm_embedding_dim if args.encoding_type == "sas" else None,
                },
                model_save_path,
            )
            print(f"Model saved to {model_save_path} (Val Loss: {best_val_loss:.4f})")
    print("LSTM Training finished.")


def main():
    print("Starting unified training script for PaTS models...")

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
    parser.add_argument(
        "--encoding_type",
        type=str,
        default="bin",
        choices=["bin", "sas"],
        help="The encoding type of the dataset to use.",
    )

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
    parser.add_argument("--use_mlm_task", action="store_true", help="Enable MLM auxiliary task for LSTM.")
    parser.add_argument("--mlm_loss_weight", type=float, default=0.2, help="Weight for the MLM auxiliary loss.")
    parser.add_argument(
        "--mlm_mask_prob", type=float, default=0.15, help="Probability of masking a predicate for the MLM task."
    )
    parser.add_argument("--lstm_embedding_dim", type=int, default=32, help="Embedding dimension for SAS+ encoding.")

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

    parser.add_argument(
        "--use_constraint_loss", action="store_true", help="Enable constraint violation auxiliary loss for LSTM."
    )
    parser.add_argument(
        "--constraint_loss_weight", type=float, default=1.0, help="Weight for the constraint violation auxiliary loss."
    )

    args = parser.parse_args()

    print("Parsed Arguments:")
    pprint(vars(args))

    if args.model_type == "ttm" and (args.mlm_loss_weight or args.mlm_mask_prob):
        warnings.warn("MLM-related arguments are not applicable for TTM. Ignoring them.")

    if args.model_type == "ttm" and (args.use_constraint_loss or args.constraint_loss_weight != 1.0):
        warnings.warn("Constraint loss arguments are not applicable for TTM. Ignoring them.")

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

    # Instantiate the validator for use in training
    validator = None
    if args.model_type == "lstm" and args.use_constraint_loss:
        print("Constraint loss enabled. Initializing BlocksWorldValidator...")
        if args.encoding_type == "bin":
            manifest_path = args.dataset_dir / f"predicate_manifest_{args.num_blocks}.txt"
            if not manifest_path.exists():
                print(
                    f"ERROR: Predicate manifest not found at {manifest_path}. Cannot use constraint loss for binary encoding."
                )
                sys.exit(1)
            try:
                validator = BlocksWorldValidator(args.num_blocks, args.encoding_type, predicate_manifest_file=manifest_path)
                print("Validator for binary encoding initialized successfully.")
            except Exception as e:
                print(f"ERROR: Failed to initialize validator: {e}")
                sys.exit(1)
        elif args.encoding_type == "sas":
            # SAS validator doesn't need a manifest file.
            print("WARNING: Constraint loss for SAS encoding is not implemented and will have no effect.")
            validator = BlocksWorldValidator(args.num_blocks, args.encoding_type)
            print("Validator for SAS encoding initialized.")

    # Load Datasets
    print("Loading datasets...")
    try:
        train_dataset = PaTSDataset(
            dataset_dir=args.dataset_dir, split_file_name="train_files.txt", encoding_type=args.encoding_type
        )
        val_dataset = PaTSDataset(
            dataset_dir=args.dataset_dir, split_file_name="val_files.txt", encoding_type=args.encoding_type
        )
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
        # Use partial to create a collate function with the mlm_mask_prob argument
        collate_fn_with_args = partial(lstm_collate_fn, mlm_mask_prob=args.mlm_mask_prob if args.use_mlm_task else 0.0)

        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_with_args, num_workers=0
        )
        val_dataloader = (
            DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_with_args, num_workers=0
            )
            if len(val_dataset) > 0
            else None
        )

        model = PaTS_LSTM(
            num_features=num_features,
            hidden_size=args.lstm_hidden_size,
            num_lstm_layers=args.lstm_num_layers,
            dropout_prob=args.lstm_dropout_prob,
            use_mlm_task=args.use_mlm_task if args.encoding_type == "binary" else False,  # Disable MLM for SAS+
            encoding_type=args.encoding_type,
            num_blocks=args.num_blocks,
            embedding_dim=args.lstm_embedding_dim,
        ).to(DEVICE)
        setattr(model, "target_num_blocks", args.num_blocks)
        print(model)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable LSTM parameters: {total_params}")

        lstm_model_save_path = model_specific_output_dir / f"pats_lstm_model_N{args.num_blocks}.pth"
        train_lstm_model_loop(model, train_dataloader, val_dataloader, validator, args, num_features, lstm_model_save_path)

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

```

- `PaTS/scripts/models/ttm.py`
```py
import json
import math
import posixpath
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path, PosixPath
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from loguru import logger
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset  # Used for type hinting, PaTSDataset will be the actual one
from torch.utils.tensorboard.writer import SummaryWriter
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer import Trainer
from transformers.trainer_callback import EarlyStoppingCallback, TrainerCallback
from transformers.training_args import TrainingArguments
from tsfm_public import TrackingCallback  # type: ignore
from tsfm_public.toolkit.get_model import get_model  # type: ignore

from ..pats_dataset import PaTSDataset

# ** Constants **
DEFAULT_TTM_MODEL_PATH = "ibm-granite/granite-timeseries-ttm-r2.1"  # Default TTM model
# Define the supported models based on the provided combinations: https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2/tree/main
SUPPORTED_MODELS = [
    {"context_length": 52, "prediction_length": 16, "freq_tuning": True, "loss_metric": "mse", "release": "r2.1"},
    {"context_length": 52, "prediction_length": 16, "freq_tuning": True, "loss_metric": "mae", "release": "r2.1"},
    {"context_length": 90, "prediction_length": 30, "freq_tuning": True, "loss_metric": "mse", "release": "r2.1"},
    {"context_length": 90, "prediction_length": 30, "freq_tuning": True, "loss_metric": "mae", "release": "r2.1"},
    {"context_length": 180, "prediction_length": 60, "freq_tuning": True, "loss_metric": "mae", "release": "r2.1"},
    {"context_length": 360, "prediction_length": 60, "freq_tuning": True, "loss_metric": "mae", "release": "r2.1"},
    {"context_length": 1024, "prediction_length": 96, "freq_tuning": False, "loss_metric": "mse", "release": "r2"},
    {"context_length": 1024, "prediction_length": 192, "freq_tuning": False, "loss_metric": "mse", "release": "r2"},
    {"context_length": 1024, "prediction_length": 336, "freq_tuning": False, "loss_metric": "mse", "release": "r2"},
    {"context_length": 1024, "prediction_length": 720, "freq_tuning": False, "loss_metric": "mse", "release": "r2"},
    {"context_length": 512, "prediction_length": 48, "freq_tuning": True, "loss_metric": "mse", "release": "r2.1"},
    {"context_length": 512, "prediction_length": 48, "freq_tuning": True, "loss_metric": "mae", "release": "r2.1"},
    {"context_length": 512, "prediction_length": 96, "freq_tuning": True, "loss_metric": "mse", "release": "r2.1"},
    {"context_length": 512, "prediction_length": 96, "freq_tuning": True, "loss_metric": "mae", "release": "r2.1"},
    {"context_length": 512, "prediction_length": 192, "freq_tuning": False, "loss_metric": "mse", "release": "r2"},
    {"context_length": 512, "prediction_length": 336, "freq_tuning": False, "loss_metric": "mse", "release": "r2"},
    {"context_length": 512, "prediction_length": 720, "freq_tuning": False, "loss_metric": "mse", "release": "r2"},
    {"context_length": 1536, "prediction_length": 96, "freq_tuning": False, "loss_metric": "mse", "release": "r2"},
    {"context_length": 1536, "prediction_length": 192, "freq_tuning": False, "loss_metric": "mse", "release": "r2"},
    {"context_length": 1536, "prediction_length": 336, "freq_tuning": False, "loss_metric": "mse", "release": "r2"},
    {"context_length": 1536, "prediction_length": 720, "freq_tuning": False, "loss_metric": "mse", "release": "r2"},
]

global DEVICE
DEVICE: torch.device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
# DEVICE = torch.device("cpu")


# ** Data Classes **
@dataclass
class ModelConfig:
    context_length: int
    prediction_length: int
    learning_rate: float
    batch_size: int
    num_epochs: int
    ttm_model_path: str
    seed: int
    state_dim: Optional[int] = None  # Will be set during training or loaded


# Callback class for custom TensorBoard logging
class TensorBoardLoggingCallback(TrainerCallback):
    def __init__(self, tb_writer: SummaryWriter):
        self.tb_writer = tb_writer

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Called when logging occurs during training"""
        if logs is not None and self.tb_writer is not None:
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    # Log all numeric metrics
                    self.tb_writer.add_scalar(f"training/{key}", value, state.global_step)

    def on_evaluate(self, args, state, control, model=None, logs=None, **kwargs):
        """Called after evaluation"""
        if logs is not None and self.tb_writer is not None:
            for key, value in logs.items():
                if isinstance(value, (int, float)) and key.startswith("eval_"):
                    # Log evaluation metrics
                    self.tb_writer.add_scalar(f"evaluation/{key}", value, state.global_step)


class TTMDataCollator:
    """
    Data collator for TTM model.
    Takes a list of dictionaries from PaTSDataset (each containing 'initial_state', 'goal_state', 'expert_trajectory', 'id' as numpy arrays) and transforms them into a batch dictionary suitable for TTM input.
    This includes creating past/future values, masks, scaling, and padding.
    """

    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        state_dim: int,  # state_dim is crucial for creating correct shapes
    ):
        self.context_length: int = context_length
        self.prediction_length: int = prediction_length
        self.state_dim: int = state_dim

    def _scale_binary_array(self, data_array_np: np.ndarray) -> np.ndarray:
        """Scales a numpy array of 0s and 1s to -1s and 1s."""
        return data_array_np * 2.0 - 1.0

    def __call__(self, batch_items: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
        # batch_items is a list of dicts from PaTSDataset.__getitem__
        # Each dict: {'initial_state', 'goal_state', 'expert_trajectory', 'id'}
        # All values are np.float32 numpy arrays.

        batch_size = len(batch_items)

        list_past_values_scaled = []
        list_future_values_scaled = []
        list_past_observed_mask = []
        list_future_observed_mask = []
        list_static_categorical_scaled = []

        # print(batch_items)
        for item in batch_items:
            plan_states_np_orig = item["expert_trajectory"]  # (L, F), 0/1
            goal_state_np_orig = item["goal_state"]  # (F,), 0/1
            initial_state_np_orig = item["initial_state"]  # (F,), 0/1

            past_values_np = np.zeros((self.context_length, self.state_dim), dtype=np.float32)
            past_observed_mask_np = np.zeros((self.context_length, self.state_dim), dtype=np.float32)

            num_plan_steps_for_context = min(len(plan_states_np_orig), self.context_length)

            if num_plan_steps_for_context > 0:
                past_values_np[:num_plan_steps_for_context] = plan_states_np_orig[:num_plan_steps_for_context]
                past_observed_mask_np[:num_plan_steps_for_context, :] = 1.0
                if num_plan_steps_for_context < self.context_length:
                    last_observed_state_in_context = plan_states_np_orig[num_plan_steps_for_context - 1]
                    num_past_padding = self.context_length - num_plan_steps_for_context
                    padding_values = np.tile(last_observed_state_in_context, (num_past_padding, 1))
                    past_values_np[num_plan_steps_for_context:] = padding_values
            elif self.context_length > 0:  # plan_states_np_orig might be empty or shorter than context
                padding_values = np.tile(initial_state_np_orig, (self.context_length, 1))
                past_values_np[:] = padding_values

            future_values_np = np.zeros((self.prediction_length, self.state_dim), dtype=np.float32)
            future_observed_mask_np = np.zeros((self.prediction_length, self.state_dim), dtype=np.float32)

            target_future_plan_states_orig_list = []
            if len(plan_states_np_orig) > self.context_length:
                target_future_plan_states_orig_list = plan_states_np_orig[self.context_length :]

            target_future_plan_states_orig = np.array(target_future_plan_states_orig_list, dtype=np.float32)
            num_actual_future_steps_from_plan = (
                target_future_plan_states_orig.shape[0] if target_future_plan_states_orig.ndim == 2 else 0
            )
            len_to_copy_to_future = min(num_actual_future_steps_from_plan, self.prediction_length)

            if len_to_copy_to_future > 0:
                future_values_np[:len_to_copy_to_future] = target_future_plan_states_orig[:len_to_copy_to_future]
                future_observed_mask_np[:len_to_copy_to_future, :] = 1.0

            if len_to_copy_to_future < self.prediction_length:
                num_future_padding = self.prediction_length - len_to_copy_to_future
                padding_values = np.tile(goal_state_np_orig, (num_future_padding, 1))
                future_values_np[len_to_copy_to_future:] = padding_values
                future_observed_mask_np[len_to_copy_to_future:, :] = 1.0

            static_categorical_values_np = goal_state_np_orig.copy()

            list_past_values_scaled.append(self._scale_binary_array(past_values_np))
            list_future_values_scaled.append(self._scale_binary_array(future_values_np))
            list_past_observed_mask.append(past_observed_mask_np)
            list_future_observed_mask.append(future_observed_mask_np)
            list_static_categorical_scaled.append(self._scale_binary_array(static_categorical_values_np))

        # Stack and convert to tensors. Trainer will move to device.
        return {
            "freq_token": torch.zeros(batch_size, dtype=torch.long),
            "past_values": torch.from_numpy(np.array(list_past_values_scaled, dtype=np.float32)),
            "future_values": torch.from_numpy(np.array(list_future_values_scaled, dtype=np.float32)),
            "past_observed_mask": torch.from_numpy(np.array(list_past_observed_mask, dtype=np.float32)),
            "future_observed_mask": torch.from_numpy(np.array(list_future_observed_mask, dtype=np.float32)),
            "static_categorical_values": torch.from_numpy(np.array(list_static_categorical_scaled, dtype=np.float32)),
        }


# ** Model Class **
class BlocksWorldTTM:
    def __init__(self, model_config: ModelConfig, device: torch.device, output_dir: PosixPath | Path):
        self.config = model_config
        self.device: torch.device = device
        self.model: PreTrainedModel
        self.model_name: str
        self.trainer: Trainer
        self.output_dir: PosixPath | Path = output_dir
        self.tb_writer: Optional[SummaryWriter] = None

    def train(
        self,
        train_dataset,
        val_dataset: Optional[Dataset] = None,
    ):
        logger.info("Starting model training...")
        logger.info(f"Initializing TTM model from: {self.config.ttm_model_path}")

        get_model_params = {
            "model_path": self.config.ttm_model_path,
            "context_length": self.config.context_length,
            "prediction_length": self.config.prediction_length,
            "head_dropout": 0.1,
        }
        self.model: PreTrainedModel = get_model(**get_model_params).to(self.device)  # type: ignore

        # Find model name key for logging
        get_model_params["return_model_key"] = True
        self.model_name = str(get_model(**get_model_params))  # Ensure it's string
        logger.info(f"Base TTM model key: {self.model_name}")

        # Initialize TensorBoard writer
        tensorboard_log_dir = self.output_dir / "tensorboard_logs"
        tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=str(tensorboard_log_dir))
        logger.info(f"TensorBoard logs will be saved to: {tensorboard_log_dir}")

        training_args = TrainingArguments(
            output_dir=posixpath.join(self.output_dir, "training_output"),
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            eval_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            seed=self.config.seed,
            report_to="tensorboard",  # Enable TensorBoard reporting
            logging_dir=str(tensorboard_log_dir),  # Set logging directory
            logging_strategy="steps",
            logging_steps=10,  # Log every 10 steps
            dataloader_pin_memory=False,
            remove_unused_columns=False,  # Do not remove unused columns
        )

        # Callbacks
        callbacks = [
            TrackingCallback(),
            EarlyStoppingCallback(early_stopping_patience=5),
            TensorBoardLoggingCallback(self.tb_writer),  # Custom callback for additional logging
        ]

        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate)

        # Calculate steps_per_epoch carefully, especially with SubsetRandomSampler
        num_train_samples = len(train_dataset)
        steps_per_epoch = math.ceil(num_train_samples / (self.config.batch_size * training_args.world_size))

        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.config.learning_rate,
            epochs=self.config.num_epochs,
            steps_per_epoch=steps_per_epoch,
        )

        # Log hyperparameters to TensorBoard
        self.tb_writer.add_hparams(
            {
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "num_epochs": self.config.num_epochs,
                "context_length": self.config.context_length,
                "prediction_length": self.config.prediction_length,
                "state_dim": self.config.state_dim,
            },
            {},
        )

        # Instantiate the data collator
        ttm_collator = TTMDataCollator(
            context_length=self.config.context_length,
            prediction_length=self.config.prediction_length,
            state_dim=self.config.state_dim,  # type: ignore
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=callbacks,
            data_collator=ttm_collator,  # Use the custom collator
            optimizers=(optimizer, scheduler),  # type: ignore
        )

        logger.info("Trainer initialized. Starting training...")
        self.trainer.train()
        logger.info("Training finished.")

        # Close TensorBoard writer
        if self.tb_writer:
            self.tb_writer.close()

    def predict(self, context_sequence: torch.Tensor, goal_states: torch.Tensor) -> torch.Tensor:
        """
        Predicts a sequence of future states given a context sequence and goal states.
        This is the core model call used by the autoregressive loop.

        :param context_sequence: A tensor of shape (batch_size, context_length, num_features) containing the recent history of states (in 0/1 format).
        :param goal_states: A tensor of shape (batch_size, num_features) for the goal states (in 0/1 format).
        :return: A tensor of shape (batch_size, prediction_length, num_features) containing the predicted future states (in 0/1 format).
        """
        if self.model is None:
            raise RuntimeError("Model needs to be trained or loaded before prediction.")
        if self.config.state_dim is None:
            raise RuntimeError("Model config state_dim is not set.")

        self.model.eval()
        with torch.no_grad():
            batch_size = context_sequence.shape[0]

            # Scale inputs from 0/1 to -1/1 for the model
            context_sequence_scaled = context_sequence.to(self.device) * 2.0 - 1.0
            goal_states_scaled = goal_states.to(self.device) * 2.0 - 1.0

            inputs = {
                "past_values": context_sequence_scaled,
                "past_observed_mask": torch.ones_like(context_sequence_scaled).to(self.device),
                "static_categorical_values": goal_states_scaled,
                "freq_token": torch.zeros(batch_size, dtype=torch.long).to(self.device),
            }

            outputs = self.model(**inputs)
            raw_logits = outputs[0]  # Shape: (batch_size, prediction_length, num_features)

            # Binarize the output logits
            predictions_tanh = torch.tanh(raw_logits)
            predictions_scaled_binary = torch.where(
                predictions_tanh > 0, torch.tensor(1.0, device=self.device), torch.tensor(-1.0, device=self.device)
            )

            # Convert back from -1/1 to 0/1 for the return value
            predictions_original_binary = (predictions_scaled_binary + 1.0) / 2.0

        return predictions_original_binary

    def save(self, path: Path):
        """Save model weights and configuration"""
        if self.model is None:
            raise RuntimeError("No model to save. Train or load a model first.")

        path.mkdir(parents=True, exist_ok=True)
        model_path = path / "model.pt"
        torch.save(self.model.state_dict(), model_path)

        config_path = path / "config.json"
        with open(config_path, "w") as f:
            json.dump(asdict(self.config), f, indent=4)

        logger.info(f"Model and config saved to {path}")

    @classmethod
    def load(cls, path: Path, device: torch.device) -> "BlocksWorldTTM":
        """Load model weights and configuration"""
        logger.info(f"Loading model from {path}")
        config_path = path / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config_dict = json.load(f)

        # Ensure all required fields for ModelConfig are present
        try:
            loaded_model_config = ModelConfig(**config_dict)
        except TypeError as e:
            logger.error(f"Error loading ModelConfig from {config_path}. Missing fields or mismatch: {e}")
            logger.error(f"Loaded config_dict: {config_dict}")
            raise

        if loaded_model_config.state_dim is None:
            raise ValueError("Loaded model config must contain 'state_dim'.")

        instance = cls(model_config=loaded_model_config, device=device, output_dir=path)

        get_model_params = {
            "model_path": loaded_model_config.ttm_model_path,
            "context_length": loaded_model_config.context_length,
            "prediction_length": loaded_model_config.prediction_length,
            "head_dropout": 0.1,
        }
        instance.model = get_model(**get_model_params).to(instance.device)  # type: ignore

        model_path = path / "model.pt"
        if not model_path.is_file():
            raise FileNotFoundError(f"Model weights file not found: {model_path}")

        instance.model.load_state_dict(torch.load(model_path, map_location=instance.device))
        instance.model.eval()

        # Try to get model_name
        try:
            get_model_params["return_model_key"] = True
            instance.model_name = str(get_model(**get_model_params))
        except Exception:
            instance.model_name = Path(instance.config.ttm_model_path).name

        logger.info(f"Model loaded successfully from {path}. Base TTM: {instance.model_name}")
        return instance


# ** Helper Functions **
def setup_logging(level="INFO", log_file: Optional[Path] = None):
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        level=level.upper(),
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            rotation="10 MB",
            level="DEBUG",  # Log DEBUG to file
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        )
    logger.info(f"Logging setup complete. Level: {level}. File: {log_file}")


def load_problem_basenames_from_split_file(split_file_path: Path) -> List[str]:
    if not split_file_path.exists():
        logger.warning(f"Split file not found: {split_file_path}")
        return []
    with open(split_file_path, "r") as f:
        basenames = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return basenames


def get_model_path(model_config: Dict[str, Union[int, bool, str]]) -> str:
    """Constructs the model path from its configuration."""
    parts = ["ibm-granite/granite-timeseries-ttm-r2"]  # Base path
    config_str = f"{model_config['context_length']}-{model_config['prediction_length']}"
    if model_config["freq_tuning"]:
        config_str += "-ft"
    if model_config["loss_metric"] == "mae":
        # Note: Using 'l1' as it appears in the user-provided list.
        config_str += "-l1"
    config_str += f"-{model_config['release']}"
    return f"{parts[0]}/{config_str}"


def determine_ttm_model(
    max_plan_length: int,
    recommended_prediction_length: int,
    user_context_length: Optional[int] = None,
    user_prediction_length: Optional[int] = None,
    user_freq_tuning: Optional[bool] = None,
    user_loss_metric: Optional[str] = None,
    user_release: Optional[str] = None,
) -> Optional[Dict[str, Union[int, bool, str]]]:
    """
    Determines the optimal TTM model based on constraints and user preferences.
    """
    candidates = SUPPORTED_MODELS

    # 1. Filter based on user overrides
    if user_context_length is not None:
        candidates = [m for m in candidates if m["context_length"] == user_context_length]
    if user_prediction_length is not None:
        candidates = [m for m in candidates if m["prediction_length"] == user_prediction_length]
    if user_freq_tuning is not None:
        candidates = [m for m in candidates if m["freq_tuning"] == user_freq_tuning]
    if user_loss_metric is not None:
        candidates = [m for m in candidates if m["loss_metric"] == user_loss_metric]
    if user_release is not None:
        candidates = [m for m in candidates if m["release"] == user_release]

    if not candidates:
        logger.warning("No models match the specified user overrides.")
        return None

    # 2. Filter by max_plan_length
    valid_candidates = [m for m in candidates if m["context_length"] <= max_plan_length]
    if not valid_candidates:
        min_supported_cl = min(m["context_length"] for m in candidates)
        logger.warning(
            f"Max plan length ({max_plan_length}) is smaller than all supported context lengths for the current selection. "
            f"Considering models with the smallest supported context length: {min_supported_cl}."
        )
        # Relax the constraint to the smallest possible context length among the candidates
        valid_candidates = [m for m in candidates if m["context_length"] == min_supported_cl]

    # 3. Find the best prediction_length
    # Find models with prediction length <= recommended
    pl_candidates = [m for m in valid_candidates if m["prediction_length"] <= recommended_prediction_length]

    if pl_candidates:
        # If there are such models, find the largest prediction length among them
        best_pl = max(m["prediction_length"] for m in pl_candidates)
        final_candidates = [m for m in pl_candidates if m["prediction_length"] == best_pl]
    else:
        # Otherwise, choose the smallest available prediction length
        best_pl = min(m["prediction_length"] for m in valid_candidates)
        logger.warning(
            f"Recommended prediction length ({recommended_prediction_length}) is smaller than all supported forecast lengths. "
            f"Choosing the smallest supported: {best_pl}."
        )
        final_candidates = [m for m in valid_candidates if m["prediction_length"] == best_pl]

    logger.info(
        f"Recommended prediction length: {recommended_prediction_length} | Auto-selected prediction_length: {best_pl}"
    )

    # 4. Select the one with the largest context_length from the finalists
    best_cl = max(m["context_length"] for m in final_candidates)
    final_candidates = [m for m in final_candidates if m["context_length"] == best_cl]
    logger.info(f"Max plan length: {max_plan_length} | Auto-selected context_length: {best_cl}")

    # 5. Tie-break if multiple models remain
    # Sort by: freq_tuning (True first), loss_metric ('mae' first), release (e.g., 'r2.1' > 'r2')
    final_candidates.sort(key=lambda m: (not m["freq_tuning"], m["loss_metric"] != "mae", m["release"]), reverse=True)

    selected_model = final_candidates[0]

    logger.info(f"Selected model configuration: {selected_model}")
    logger.info(f"Model path: {get_model_path(selected_model)}")

    return selected_model


def get_num_blocks_from_filename(filename_base: str) -> Optional[int]:
    """Extracts number of blocks from a filename like 'blocks_3_problem_1'."""
    match = re.search(r"blocks_(\d+)_problem", filename_base)
    if match:
        return int(match.group(1))
    return None


def prepare_datasets(
    dataset_dir: Path, dataset_split_dir: Path, num_blocks: int, context_length: int, prediction_length: int, seed: int
) -> Tuple[Dataset, Dataset, Dataset, int, int]:
    logger.info("Preparing train/validation/test datasets...")

    train_basenames_all = load_problem_basenames_from_split_file(dataset_split_dir / "train_files.txt")
    val_basenames_all = load_problem_basenames_from_split_file(dataset_split_dir / "val_files.txt")
    # Test dataset is not strictly needed for training, but can be prepared for completeness
    # test_basenames_all = load_problem_basenames_from_split_file(dataset_split_dir / "test_files.txt")

    # Filter basenames for the target num_blocks (important if split files are mixed)
    # Assuming dataset_dir is already blocks_<N> specific, this might be redundant but safe.
    train_basenames = [bn for bn in train_basenames_all if get_num_blocks_from_filename(bn) == num_blocks]
    val_basenames = [bn for bn in val_basenames_all if get_num_blocks_from_filename(bn) == num_blocks]
    # test_basenames = [bn for bn in test_basenames_all if get_num_blocks_from_filename(bn) == num_blocks] # Test not strictly needed for training

    if not train_basenames:
        raise ValueError(
            f"No training problem basenames found for N={num_blocks} in {dataset_split_dir / 'train_files.txt'}"
        )

    logger.info(f"Found {len(train_basenames)} train, {len(val_basenames)} val basenames for N={num_blocks}.")

    # Create dataset instances
    # PaTSDataset will infer state_dim from the data.
    # dataset_dir is e.g. data/blocks_4, split_file_name is e.g. "train_files.txt"
    train_dataset_instance = PaTSDataset(dataset_dir=dataset_dir, split_file_name="train_files.txt")
    state_dim = train_dataset_instance.state_dim
    val_dataset_instance = PaTSDataset(dataset_dir=dataset_dir, split_file_name="val_files.txt")
    test_dataset_instance = PaTSDataset(dataset_dir=dataset_dir, split_file_name="test_files.txt")

    # Get max_plan_len_in_train_data using PaTSDataset structure
    max_plan_len_in_train_data = 0
    if len(train_dataset_instance.basenames) > 0:
        for basename_for_len_check in train_dataset_instance.basenames:
            traj_file_path_for_len = train_dataset_instance.trajectories_bin_dir / f"{basename_for_len_check}.traj.bin.npy"
            if traj_file_path_for_len.exists():
                try:
                    traj_np = np.load(traj_file_path_for_len)
                    if traj_np.ndim == 2:
                        max_plan_len_in_train_data = max(max_plan_len_in_train_data, traj_np.shape[0])
                except Exception as e:
                    logger.warning(f"Could not load trajectory {traj_file_path_for_len} to determine max length: {e}")
            else:
                logger.warning(f"Trajectory file {traj_file_path_for_len} not found during max length check.")

    logger.info(
        f"Dataset preparation complete. state_dim: {state_dim}, max_plan_len_in_train_data: {max_plan_len_in_train_data}"
    )

    return train_dataset_instance, val_dataset_instance, test_dataset_instance, state_dim, max_plan_len_in_train_data

```

- `PaTS/scripts/models/lstm.py`
```py
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

# ** Configuration **
DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))


# ** Model Definition **
class PaTS_LSTM(nn.Module):
    def __init__(
        self,
        num_features,
        hidden_size,
        num_lstm_layers,
        dropout_prob=0.2,
        use_mlm_task=False,
        encoding_type="binary",
        num_blocks=None,
        embedding_dim=32,
    ):
        """
        Initializes the PaTS_LSTM model.

        :param num_features: The number of features in a state vector.
        :type num_features: int
        :param hidden_size: The size of the LSTM hidden state.
        :type hidden_size: int
        :param num_lstm_layers: The number of layers in the LSTM.
        :type num_lstm_layers: int
        :param dropout_prob: Dropout probability.
        :type dropout_prob: float
        :param use_mlm_task: If True, adds an auxiliary MLM head.
        :type use_mlm_task: bool
        :param encoding_type: The encoding type of the data ('bin' or 'sas').
        :type encoding_type: str
        :param num_blocks: The number of blocks in the SAS+ encoding (required if encoding_type is 'sas').
        :type num_blocks: int | None
        :param embedding_dim: The dimension of the embedding for SAS+ encoding.
        :type embedding_dim: int
        """
        super(PaTS_LSTM, self).__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.use_mlm_task = use_mlm_task
        self.encoding_type = encoding_type
        self.num_blocks = num_blocks
        self.embedding_dim = embedding_dim

        if self.encoding_type == "sas":
            if self.num_blocks is None:
                raise ValueError("num_blocks must be provided for SAS+ encoding.")
            # For N blocks, locations are: arm (-1), table (0), on_block_1 (1)... on_block_N (N)
            # Total N+2 locations.
            self.num_locations = self.num_blocks + 2
            # Embedding layer to convert location indices to vectors.
            # We add 1 to num_locations because SAS+ value -1 maps to index 0.
            self.location_embedding = nn.Embedding(self.num_locations, self.embedding_dim)
            # LSTM input is the concatenation of embedded current state and goal state
            lstm_input_size = 2 * self.num_blocks * self.embedding_dim
            # The head predicts a location for each block
            self.forecasting_head = nn.Linear(hidden_size, self.num_blocks * self.num_locations)
            print(
                f"INFO: PaTS_LSTM (SAS+) initialized. Num locations: {self.num_locations}, Embedding dim: {self.embedding_dim}"
            )
        else:  # Binary encoding
            lstm_input_size = 2 * num_features
            self.forecasting_head = nn.Linear(hidden_size, num_features)
            if use_mlm_task:
                self.mlm_head = nn.Linear(hidden_size, num_features)
            print("INFO: PaTS_LSTM (binary) initialized.")

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,  # Expects (batch, seq_len, features)
            dropout=dropout_prob if num_lstm_layers > 1 else 0.0,
        )

    def _map_sas_to_indices(self, sas_tensor):
        # SAS+ values: -1 (arm), 0 (table), 1..N (on block 1..N)
        # Embedding indices: 0, 1, 2..N+1
        return sas_tensor + 1

    def _map_indices_to_sas(self, indices_tensor):
        # Embedding indices -> SAS+ values
        return indices_tensor - 1

    def forward(self, current_states_batch, goal_state_batch, lengths, h_init=None, c_init=None):
        """
        Forward pass for training or multi-step inference.
        :param current_states_batch: Batch of current state sequences (B, S_max, F). Padded sequences.
        :type current_states_batch: Tensor
        :param goal_state_batch: Batch of goal states (B, F).
        :type goal_state_batch: Tensor
        :param lengths: Batch of original sequence lengths (B,). For packing.
        :type lengths: Tensor
        :param h_init: Initial hidden state.
        :type h_init: Tensor | None
        :param c_init: Initial cell state.
        :type c_init: Tensor | None
        :returns:
            - forecasting_logits: Logits for predicted next states (B, S_max, F).
            - mlm_logits: Logits for MLM state reconstruction. None if use_mlm_task is False.
            - (h_n, c_n): Last hidden and cell states.
        :rtype: Tuple[Tensor, Tensor | None, Tuple[Tensor, Tensor]]
        """
        if self.encoding_type == "sas":
            # Map SAS+ values to non-negative indices for embedding lookup
            current_indices = self._map_sas_to_indices(current_states_batch)
            goal_indices = self._map_sas_to_indices(goal_state_batch)

            # Embed the state sequences and goal states
            current_embedded = self.location_embedding(current_indices)  # (B, S_max, N, E)
            goal_embedded = self.location_embedding(goal_indices)  # (B, N, E)

            # Flatten the embeddings for LSTM input
            batch_size, max_seq_len, _, _ = current_embedded.shape
            current_flat = current_embedded.view(batch_size, max_seq_len, -1)
            goal_flat = goal_embedded.view(batch_size, -1)
            goal_expanded = goal_flat.unsqueeze(1).repeat(1, max_seq_len, 1)

            lstm_input = torch.cat((current_flat, goal_expanded), dim=2)
        else:  # Binary encoding
            batch_size, max_seq_len, _ = current_states_batch.shape
            goal_state_expanded = goal_state_batch.unsqueeze(1).repeat(1, max_seq_len, 1)
            lstm_input = torch.cat((current_states_batch, goal_state_expanded), dim=2)

        # Common LSTM Logic
        # Lengths should be on CPU for pack_padded_sequence
        packed_input = pack_padded_sequence(lstm_input, lengths.cpu(), batch_first=True, enforce_sorted=False)

        if h_init is None or c_init is None:
            h_0 = torch.zeros(self.num_lstm_layers, batch_size, self.hidden_size).to(lstm_input.device)
            c_0 = torch.zeros(self.num_lstm_layers, batch_size, self.hidden_size).to(lstm_input.device)
        else:
            h_0, c_0 = h_init, c_init

        packed_output, (h_n, c_n) = self.lstm(packed_input, (h_0, c_0))

        # Unpack sequence
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=max_seq_len)

        # Head logic
        forecasting_logits = self.forecasting_head(lstm_out)
        mlm_logits = None  # MLM not supported for SAS+ here

        if self.encoding_type == "sas":
            # Reshape logits to (B, S_max, num_blocks, num_locations) for CrossEntropyLoss
            forecasting_logits = forecasting_logits.view(batch_size, max_seq_len, self.num_blocks, self.num_locations)

        return forecasting_logits, mlm_logits, (h_n, c_n)

    def predict_step(self, current_state_S_t, goal_state_S_G, h_prev, c_prev):
        """
        Predicts the single next state for inference.
        :param current_state_S_t: Current state (1, F).
        :type current_state_S_t: Tensor
        :param goal_state_S_G: Goal state (1, F).
        :type goal_state_S_G: Tensor
        :param h_prev: Previous hidden state from LSTM.
        :type h_prev: Tensor
        :param c_prev: Previous cell state from LSTM.
        :type c_prev: Tensor
        :returns:
            - next_state_binary: Predicted binary next state (1, F).
            - next_state_probs: Predicted probabilities for next state (1, F).
            - (h_next, c_next): New hidden and cell states.
        :rtype: Tuple[Tensor, Tensor, Tuple[Tensor, Tensor]]
        """
        self.eval()  # Set to evaluation mode

        if self.encoding_type == "sas":
            current_indices = self._map_sas_to_indices(current_state_S_t).unsqueeze(1)  # (1, 1, N)
            goal_indices = self._map_sas_to_indices(goal_state_S_G)  # (1, N)

            current_embedded = self.location_embedding(current_indices)  # (1, 1, N, E)
            goal_embedded = self.location_embedding(goal_indices)  # (1, N, E)

            current_flat = current_embedded.view(1, 1, -1)
            goal_flat = goal_embedded.view(1, -1)
            goal_expanded = goal_flat.unsqueeze(1)

            lstm_input_step = torch.cat((current_flat, goal_expanded), dim=2)
        else:  # Binary
            current_state_S_t_seq = current_state_S_t.unsqueeze(1)
            goal_state_S_G_expanded = goal_state_S_G.unsqueeze(1)
            lstm_input_step = torch.cat((current_state_S_t_seq, goal_state_S_G_expanded), dim=2)

        # LSTM expects (h_0, c_0) even for a single step if states are passed
        lstm_out, (h_next, c_next) = self.lstm(lstm_input_step, (h_prev, c_prev))
        # lstm_out: (1, 1, H)

        # Use the forecasting head for prediction
        next_state_logits = self.forecasting_head(lstm_out.squeeze(1))
        if self.encoding_type == "sas":
            # Reshape to (1, num_blocks, num_locations)
            logits_per_block = next_state_logits.view(1, self.num_blocks, self.num_locations)
            # Get the predicted location index for each block
            predicted_indices = torch.argmax(logits_per_block, dim=2)  # (1, N)
            # Map back to SAS+ values
            next_state_sas = self._map_indices_to_sas(predicted_indices)
            return next_state_sas, logits_per_block, h_next, c_next
        else:  # Binary
            next_state_probs = torch.sigmoid(next_state_logits)
            next_state_binary = (next_state_probs > 0.5).float()
            return next_state_binary, next_state_probs, h_next, c_next


def lstm_collate_fn(batch, mlm_mask_prob=0.15):
    """
    Custom collate function for LSTM training.
    Handles padding and prepares data for the MLM auxiliary task.
    """
    # Filter out None items that might result from __getitem__ errors
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    input_seqs_list, target_seqs_list, goal_states_list, expert_trajectories_orig_list, ids_list = [], [], [], [], []

    for item in batch:
        # item is a dict from PaTSDataset: {'initial_state', 'goal_state', 'expert_trajectory', 'id'}
        # All values are np.float32 arrays.
        expert_trajectory_np = item["expert_trajectory"]

        if expert_trajectory_np.shape[0] <= 1:
            input_s_np = expert_trajectory_np
            target_s_np = expert_trajectory_np
        else:
            input_s_np = expert_trajectory_np[:-1, :]
            target_s_np = expert_trajectory_np[1:, :]

        input_seqs_list.append(torch.from_numpy(input_s_np))
        target_seqs_list.append(torch.from_numpy(target_s_np))
        goal_states_list.append(torch.from_numpy(item["goal_state"]))
        expert_trajectories_orig_list.append(torch.from_numpy(expert_trajectory_np))
        ids_list.append(item["id"])

    # Pad sequences
    # pad_sequence expects a list of tensors (seq_len, features)
    # and returns (max_seq_len, batch_size, features) if batch_first=False (default)
    # or (batch_size, max_seq_len, features) if batch_first=True
    lengths = torch.tensor([len(seq) for seq in input_seqs_list], dtype=torch.long)
    padded_input_seqs = pad_sequence(input_seqs_list, batch_first=True, padding_value=0.0)
    padded_target_seqs = pad_sequence(target_seqs_list, batch_first=True, padding_value=0.0)
    goal_states_batch = torch.stack(goal_states_list)

    # Create MLM Predicate Mask
    # This mask indicates which elements of the *input* sequence should be predicted by the MLM head.
    mlm_predicate_mask = torch.zeros_like(padded_input_seqs, dtype=torch.float32)
    if mlm_mask_prob > 0:
        for i in range(padded_input_seqs.shape[0]):
            seq_len = int(lengths[i])
            if seq_len > 0:
                prob_matrix = torch.full((seq_len, padded_input_seqs.shape[2]), mlm_mask_prob)
                masked_indices = torch.bernoulli(prob_matrix).bool()
                mlm_predicate_mask[i, :seq_len, :] = masked_indices.float()

    return {
        "input_sequences": padded_input_seqs,
        "goal_states": goal_states_batch,
        "target_sequences": padded_target_seqs,
        "lengths": lengths,
        "ids": ids_list,
        "expert_trajectories": expert_trajectories_orig_list,
        "mlm_predicate_mask": mlm_predicate_mask,  # Add to batch dict
    }

```

- `PaTS/pats.sh`
```sh
#! /bin/bash

#SBATCH --job-name=PaTS
#SBATCH -o r_out%j.out
#SBATCH -e r_err%j.err

#SBATCH --mail-user=niting@email.sc.edu
#SBATCH --mail-type=ALL

#SBATCH -p v100-16gb-hiprio
#SBATCH --gres=gpu:1

module load python3/anaconda/2021.07 gcc/12.2.0 cuda/12.1
source activate /home/niting/miniconda3/envs/pats-env

echo $CONDA_DEFAULT_ENV
hostname
echo "Python version: $(python --version)"

model_type='lstm'  # `lstm`, `ttm`
num_blocks=4
encoding='sas'  # `sas`, `bin`
dataset_dir="data/blocks_${num_blocks}-${encoding}"

# Generate timestamp and build unique dirs/paths
timestamp=$(date +%Y%m%d_%H%M%S)
output_dir="./training_outputs_${encoding}/${model_type}_${timestamp}"
benchmark_output_dir="./benchmark_results_${encoding}/${model_type}_${timestamp}"

if [ "$model_type" = 'lstm' ]; then
    echo "Using LSTM model"
    model_path="${output_dir}/${model_type}_N${num_blocks}/pats_lstm_model_N${num_blocks}.pth"
elif [ "$model_type" = 'ttm' ]; then
    echo "Using TTM model"
    model_path="${output_dir}/${model_type}_N${num_blocks}/final_model_assets"
else
    echo "Unsupported model type: $model_type"
    exit 1
fi

mkdir -p "$output_dir"
mkdir -p "$benchmark_output_dir"

echo -e "\n"
python -m scripts.train_model \
    --model_type $model_type \
    --dataset_dir $dataset_dir \
    --dataset_split_dir $dataset_dir \
    --num_blocks $num_blocks \
    --encoding_type $encoding \
    --output_dir $output_dir \
    --epochs 400 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --seed 13 \

echo -e "\n"
echo "Training completed. Outputs in $output_dir"

echo -e "\n"
echo "Starting benchmarking with model: $model_type"
python -m scripts.benchmark \
    --dataset_dir $dataset_dir \
    --num_blocks $num_blocks \
    --model_type $model_type \
    --model_path $model_path \
    --output_dir $benchmark_output_dir \
    --encoding_type $encoding \
    --max_plan_length 60 \
    --save_detailed_results

echo -e "\n"
echo "Benchmarking completed. Results in $benchmark_output_dir"

```

