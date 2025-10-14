# PaTS Models

> [!WARNING]
> This README is out-of-date since it doesn't reflect the newer grippers integration.

This directory contains the implementations and wrappers for various time-series models adapted to the Planning as Time-Series (PaTS) framework. Each model is designed to learn an implicit transition function from demonstrated plan trajectories and generate sequences of states (plans) autoregressively.

All models adhere to the `PlannableModel` abstract interface, which standardizes how they are loaded, how they predict sequences, and how their metadata (like `model_name` and `state_dim`) is accessed. This allows the `benchmark.py` script to evaluate different models interchangeably.

## `PlannableModel` Interface (`PlannableModel.py`)

This abstract base class defines the contract for any model that can be used within the PaTS framework. It ensures that all models provide:

- `load_model()`: To load model weights and configuration.
- `predict_sequence(initial_state_np, goal_state_np, max_length)`: The core method for generating a plan (sequence of states).
- `model_name`: A property returning a descriptive name for the model.
- `state_dim`: A property returning the feature dimension of the states the model operates on.

## Model Implementations

### 1. LSTM (`lstm.py`)

- **Core Idea**: A Recurrent Neural Network (RNN) architecture, specifically Long Short-Term Memory (LSTM), which is well-suited for learning patterns in sequential data.
- **Adaptation for PaTS**: The LSTM model takes the current state and the goal state as input at each time step and predicts the next state in the sequence. This process is repeated autoregressively until the goal is reached or a maximum plan length is exceeded.
- **Encoding Handling**:
  - **Binary Encoding (`bin`)**: The model directly processes binary state vectors. The output layer uses a sigmoid activation to predict probabilities for each predicate, which are then binarized (e.g., >0.5 -> 1).
  - **SAS+ Encoding (`sas`)**: For SAS+ (integer-based position vectors), an embedding layer is used to convert each block's position index into a dense vector representation. The LSTM processes these embeddings, and the output layer performs a multi-class classification for each block, predicting its next location (e.g., on table, on block B, held).
- **Key Parameters**: `hidden_size`, `num_lstm_layers`, `dropout_prob`, and `embedding_dim` (for SAS+).
- **Training**: Uses `BCEWithLogitsLoss` for binary encoding and `CrossEntropyLoss` for SAS+ encoding.

### 2. Tiny Time Mixer (TTM) (`ttm.py`)

- **Core Idea**: A modern, compact MLP-based architecture for time-series forecasting, leveraging the `tsfm_public` library (IBM Granite Time Series Foundation Models). TTM models are typically pre-trained on large datasets and fine-tuned for specific tasks.
- **Adaptation for PaTS**: TTM models are designed to predict a fixed `prediction_length` of future steps given a `context_length` of past steps. For PaTS, this is adapted into an autoregressive loop:
  1.  The model is given a `context_length` window of past states (initially padded with the initial state) and the goal state.
  2.  It predicts a `prediction_length` sequence of future states.
  3.  Only the first predicted state from this sequence is taken as the next step in the plan.
  4.  The context window is then updated by shifting it and appending the newly predicted state.
  5.  This process repeats until the goal is reached or `max_length` is hit.
- **`TTMDataCollator`**: A custom data collator is implemented to prepare the PaTS dataset for TTM training. It handles:
  - Creating `past_values` (context) and `future_values` (targets) from expert trajectories.
  - Padding sequences to `context_length` and `prediction_length`.
  - Scaling numerical values (e.g., 0/1 to -1/1 for binary, or just casting to float for SAS+).
  - Incorporating the goal state as `static_categorical_values`.
- **Model Selection**: The `determine_ttm_model` function helps select an appropriate pre-trained TTM variant from the `ibm-granite/granite-timeseries-ttm-r2` collection based on the dataset's `max_plan_length` and user preferences for `context_length` and `prediction_length`.

### 3. XGBoost (`xgboost.py`)

- **Core Idea**: XGBoost (eXtreme Gradient Boosting) is an ensemble learning method that uses gradient-boosted decision trees. It's known for its speed and performance on tabular data.
- **Adaptation for PaTS**: Unlike the autoregressive neural models, the XGBoost planner is trained to predict the _entire remaining plan_ in a single shot.
  1.  The input features (`X`) are constructed by concatenating a `context_window_size` of past states (e.g., `S_{t-2}, S_{t-1}, S_t`) with the `goal_state`.
  2.  The target (`y`) is the flattened sequence of all subsequent states in the expert trajectory (`S_{t+1}, S_{t+2}, ..., S_L`).
  3.  During inference, given an initial state and goal, the model predicts a long, flattened vector representing the entire plan. This vector is then reshaped back into a sequence of states.
- **Input/Output**: Data is prepared in a tabular format (`X_train`, `y_train`). The model uses `MultiOutputRegressor` to handle the multi-dimensional output.
- **Prediction**: The model outputs floating-point predictions, which are then rounded to the nearest integer and clamped to the valid range for the respective encoding (e.g., 0/1 for binary, -1 to `num_blocks` for SAS+).
- **Key Parameters**: `context_window_size` (to include past states in the input feature vector).

### 4. Llama (`llama.py`)

- **Core Idea**: Large Language Models (LLMs) like Llama are powerful generative models trained on vast amounts of text data. They can perform various tasks through prompt engineering.
- **Adaptation for PaTS**: Llama is used for zero-shot or few-shot inference. It does not undergo specific training for PaTS. Instead, it's prompted to act as a planning AI:
  1.  **Prompt Engineering**: A carefully crafted prompt instructs Llama to predict only the `NEXT_STATE` given an `INITIAL_STATE` and `GOAL_STATE`, all represented as numerical vectors in a specific string format (e.g., `"[0 1 0 0 1]"`).
  2.  **Autoregressive Loop**: Similar to LSTM/TTM, Llama is queried repeatedly. The predicted `NEXT_STATE` becomes the `INITIAL_STATE` for the subsequent query, forming an autoregressive planning process.
  3.  **Few-shot Learning**: Optionally, a single example problem-solution pair from the training data can be included in the prompt to guide Llama's generation (few-shot inference).
- **Encoding Handling**: Numerical state vectors are converted to string representations for the prompt and parsed back from Llama's text output. Robust parsing logic is crucial to handle potential malformed outputs from the LLM.
- **Limitations**: Performance is highly dependent on prompt quality and Llama's ability to generalize from numerical patterns in text. Parsing errors or non-deterministic outputs can lead to invalid plans.
