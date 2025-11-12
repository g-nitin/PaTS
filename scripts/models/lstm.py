import warnings
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


class PaTS_LSTM(nn.Module):
    def __init__(
        self,
        num_features,
        hidden_size,
        num_lstm_layers,
        dropout_prob=0.2,
        encoding_type="binary",
        domain="blocksworld",
        domain_config=None,
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
        :param encoding_type: The encoding type of the data ('bin' or 'sas').
        :type encoding_type: str
        :param domain: The planning domain ('blocksworld' or 'grippers').
        :type domain: str
        :param domain_config: Domain-specific configuration dictionary.
        :type domain_config: dict | None
        :param embedding_dim: The dimension of the embedding for SAS+ encoding.
        :type embedding_dim: int
        """
        super(PaTS_LSTM, self).__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.encoding_type = encoding_type
        self.domain = domain
        self.domain_config = domain_config or {}
        self.embedding_dim = embedding_dim

        if self.encoding_type == "sas":
            if self.domain == "blocksworld":
                self.num_blocks = self.domain_config.get("num_blocks")
                if self.num_blocks is None:
                    raise ValueError("domain_config must contain 'num_blocks' for blocksworld.")
                # For N blocks, locations are: arm (-1), table (0), on_block_1 (1)... on_block_N (N)
                self.num_locations = self.num_blocks + 2  # This is the vocabulary size per block
                self.sas_feature_dim = self.num_blocks
                print(f"INFO: PaTS_LSTM (SAS+, blocksworld) initialized. Num locations: {self.num_locations}")

            elif self.domain == "grippers":
                self.num_robots: int = self.domain_config.get("robots", 0)
                self.num_objects: int = self.domain_config.get("objects", 0)
                self.num_rooms: int = self.domain_config.get("rooms", 0)
                if any(x == 0 for x in [self.num_robots, self.num_objects, self.num_rooms]):
                    raise ValueError("domain_config must contain 'robots', 'objects', 'rooms' for grippers.")
                # Vocab: rooms (1..num_rooms) + grippers (num_rooms+1..num_rooms+2*num_robots) + padding (0)
                self.num_locations = self.num_rooms + (2 * self.num_robots) + 1
                self.sas_feature_dim = self.num_robots + self.num_objects
                print(f"INFO: PaTS_LSTM (SAS+, grippers) initialized. Vocab size: {self.num_locations}")
            else:
                raise ValueError(f"Unsupported domain for SAS+ encoding: {self.domain}")

            # Embedding layer to convert location indices to vectors.
            self.location_embedding = nn.Embedding(self.num_locations, self.embedding_dim)
            # LSTM input is the concatenation of embedded current state and goal state
            lstm_input_size = 2 * self.sas_feature_dim * self.embedding_dim
            # The head predicts a location for each feature in the SAS vector
            self.forecasting_head = nn.Linear(hidden_size, self.sas_feature_dim * self.num_locations)

        else:  # Binary encoding
            lstm_input_size = 2 * num_features
            self.forecasting_head = nn.Linear(hidden_size, num_features)
            print("INFO: PaTS_LSTM (binary) initialized.")

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,  # Expects (batch, seq_len, features)
            dropout=dropout_prob if num_lstm_layers > 1 else 0.0,
        )

    def _map_sas_to_indices(self, sas_tensor):
        if self.domain == "blocksworld":
            # SAS+ values: -1 (arm), 0 (table), 1..N (on block 1..N)
            # Embedding indices: 0, 1, 2..N+1
            return sas_tensor + 1
        elif self.domain == "grippers":
            # Robot pos: 1..num_rooms -> 1..num_rooms
            # Object pos: 1..num_rooms -> 1..num_rooms
            # Object held: -1..-2*num_robots -> num_rooms+1..num_rooms+2*num_robots
            indices = sas_tensor.clone().long()
            # Positive values (rooms) map directly to their own index
            # Negative values (grippers) are shifted into the positive range after rooms
            gripper_mask = sas_tensor < 0
            indices[gripper_mask] = self.num_rooms + torch.abs(sas_tensor[gripper_mask])
            return indices
        raise NotImplementedError(f"SAS mapping not implemented for domain: {self.domain}")

    def _map_indices_to_sas(self, indices_tensor):
        if self.domain == "blocksworld":
            # Embedding indices -> SAS+ values
            return indices_tensor - 1
        elif self.domain == "grippers":
            sas = indices_tensor.clone().long()
            # Indices > num_rooms represent grippers
            gripper_mask = indices_tensor > self.num_rooms
            # Map them back to negative values
            sas[gripper_mask] = -(indices_tensor[gripper_mask] - self.num_rooms)
            return sas
        raise NotImplementedError(f"SAS mapping not implemented for domain: {self.domain}")

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
        if self.encoding_type == "sas":
            # Reshape logits to (B, S_max, num_blocks, num_locations) for CrossEntropyLoss
            # forecasting_logits = forecasting_logits.view(batch_size, max_seq_len, self.num_blocks, self.num_locations)
            forecasting_logits = forecasting_logits.view(batch_size, max_seq_len, self.sas_feature_dim, self.num_locations)

        return forecasting_logits, (h_n, c_n)

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
            # logits_per_block = next_state_logits.view(1, self.num_blocks, self.num_locations)
            logits_per_block = next_state_logits.view(1, self.sas_feature_dim, self.num_locations)
            # Get the predicted location index for each block
            predicted_indices = torch.argmax(logits_per_block, dim=2)  # (1, N)
            # Map back to SAS+ values
            next_state_sas = self._map_indices_to_sas(predicted_indices)
            return next_state_sas, logits_per_block, h_next, c_next
        else:  # Binary
            next_state_probs = torch.sigmoid(next_state_logits)
            next_state_binary = (next_state_probs > 0.5).float()
            return next_state_binary, next_state_probs, h_next, c_next


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_forecast_losses: List[float],
    val_forecast_losses: List[float],
    epochs: int,
    model_type: str,
    encoding_type: str,
    domain_config_name: str,
    output_dir: Path,
):
    """
    Plots and saves the training and validation loss curves.
    """
    if not train_losses or not val_losses:
        print("No loss data to plot.")
        return

    sns.set_theme()  # Apply seaborn theme for aesthetics

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, epochs + 1), train_losses, label="Train Total Loss")
    plt.plot(range(1, epochs + 1), val_losses, label="Validation Total Loss")
    plt.plot(range(1, epochs + 1), train_forecast_losses, label="Train Forecast Loss", linestyle="--")
    plt.plot(range(1, epochs + 1), val_forecast_losses, label="Validation Forecast Loss", linestyle="--")

    plt.title(f"{model_type} Training & Validation Loss ({domain_config_name}, Encoding={encoding_type})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plot_filename = f"training_loss_{model_type}_{domain_config_name}_{encoding_type}.png"
    plt.savefig(output_dir / plot_filename)
    plt.close()  # Close the plot to free memory
    print(f"Training loss plot saved to {output_dir / plot_filename}")


def train_lstm_model_loop(
    model,
    train_loader,
    val_loader,
    args,
    domain_config,
    num_features,
    config_name,
    model_save_path,
    DEVICE=torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")),
):
    """
    Train the provided LSTM-based PaTS model over multiple epochs.
    This function runs a full training loop over the given `train_loader`, evaluates on `val_loader` each epoch,
    and saves the best model (by validation loss) to `model_save_path`.
    It supports two encoding modes:
        - "sas" (SAS+): forecasting is performed as a multi-class classification over locations for each block. CrossEntropyLoss is used and MLM / constraint losses are disabled.
        - "bin": forecasting is performed as independent binary predictions per block-location via BCEWithLogitsLoss. Similar to One-hot encoding. Optional MLM and constraint violation losses may be included.
    Uses AdamW optimizer and ReduceLROnPlateau scheduler (monitoring validation loss).

    - For binary encoding:
        - Forecasting loss is computed per-element using BCEWithLogitsLoss with reduction="none" and then averaged over only the valid (un-padded) forecast elements.
        - MLM loss (if args.use_mlm_task) is computed similarly by masking to the mlm_predicate_mask and averaged over masked elements.
        - Forecasting logits and targets are expected to have shape (B, S_max, N_blocks) (logits are raw, targets are 0/1 floats or ints). mlm_logits and mlm_predicate_mask (if used) should align in shape with input_seqs (B, S_max, N_blocks).
    - For SAS+ encoding:
        - Forecasting logits are expected to have shape (B, S_max, N_blocks, N_locs) and targets shape
            (B, S_max, N_blocks) containing integer class labels. Targets are mapped to class indices via model._map_sas_to_indices before applying CrossEntropyLoss.

    - Saves a checkpoint dict containing model/optimizer state_dict, training hyperparameters and metadata whenever a new best validation loss is observed.

    :param model: The PaTS LSTM model to train. Must implement forward(input_seqs, goal_states, lengths) and return a tuple (forecasting_logits, mlm_logits, ...).
        For SAS+ encoding, model must provide model.num_locations and method model._map_sas_to_indices(target_tensor) that maps SAS+ targets to class indices.
    :param train_loader: DataLoader or iterable yielding batches as dictionaries with keys:
        - "input_sequences": tensor, shape (B, S_max, N_blocks) (int for SAS+, float/binary for binary)
        - "goal_states": tensor, shape compatible with model (commonly (B, N_blocks) or (B, S_max, N_blocks))
        - "target_sequences": tensor, shape (B, S_max, N_blocks)
    :param val_loader: Validation DataLoader/iterable with the same batch dictionary format as train_loader. If None, validation is skipped (but scheduler.step will still be called with inf).
    :param args: Configuration with fields (used fields):
        - encoding_type: "sas" or "binary"
        - num_blocks: int (BlocksWorld: used for SAS+ clamping / metadata)
        - epochs: int
        - learning_rate: float
        - lstm_hidden_size, lstm_num_layers, lstm_dropout_prob, lstm_embedding_dim: training metadata saved in checkpoint
    :param domain_config: dict (Grippers: used for SAS+ clamping / metadata)
    :param num_features: Number of input features (saved in the checkpoint metadata).
    :param model_save_path: File path where the best model checkpoint dict will be saved via torch.save when validation loss improves.
    :return: None
    """

    print("\nStarting LSTM training...")

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

    # Lists to store losses for plotting
    train_losses_history = []
    val_losses_history = []
    train_forecast_losses_history = []
    val_forecast_losses_history = []

    for epoch in range(args.epochs):
        model.train()
        epoch_train_loss, epoch_forecast_loss = 0.0, 0.0
        num_train_batches = 0

        for batch_data in train_loader:
            if batch_data is None:
                continue
            input_seqs = batch_data["input_sequences"].to(DEVICE)
            goal_states = batch_data["goal_states"].to(DEVICE)
            target_seqs = batch_data["target_sequences"].to(DEVICE)
            lengths = batch_data["lengths"]

            # Safeguard for SAS+ encoding to prevent out-of-bounds embedding errors.
            if args.encoding_type == "sas":
                if model.domain == "blocksworld":
                    max_val = model.num_blocks
                    min_val = -1  # -1 for 'held', 0 for 'on table', 1..N for 'on block'
                elif model.domain == "grippers":
                    max_val = model.num_rooms
                    # Gripper values are negative, from -1 down to -(2*num_robots)
                    min_val = -(2 * model.num_robots)
                else:
                    raise ValueError(f"Unsupported domain {model.domain} for SAS+ clamping.")

                # Check if any values are out of the expected [min_val, max_val] range.
                if not sas_clamp_warning_issued and (
                    torch.any(input_seqs < min_val)
                    or torch.any(input_seqs > max_val)
                    or torch.any(goal_states < min_val)
                    or torch.any(goal_states > max_val)
                ):
                    warnings.warn(
                        f"\nTrain batch SAS+ clamping to [{min_val}, {max_val}]"
                        f"\nInput seqs: {input_seqs}"
                        f"\nSAS+ input data for domain '{model.domain}' contains values outside expected range [{min_val}, {max_val}]. "
                        f"\nClamping values to prevent embedding layer errors. "
                        f"\nPlease check your dataset for correctness."
                    )
                    sas_clamp_warning_issued = True

                input_seqs.clamp_(min=min_val, max=max_val)
                goal_states.clamp_(min=min_val, max=max_val)

            optimizer.zero_grad()
            forecasting_logits, _ = model(input_seqs, goal_states, lengths)

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
                # Create a mask for valid forecasting steps
                # (B, S_max, N_blocks, N_locs)
                forecasting_mask = torch.zeros_like(target_seqs, dtype=torch.bool).to(DEVICE)
                for i, length_val in enumerate(lengths):
                    if length_val > 0:
                        forecasting_mask[i, :length_val, :] = True  # Mark valid steps

                # Count the number of valid forecasting elements
                num_forecast_elements = forecasting_mask.float().sum()
                if num_forecast_elements == 0:
                    continue

                # Compute the loss
                loss_forecasting_unreduced = criterion(forecasting_logits, target_seqs)
                loss_forecasting = (loss_forecasting_unreduced * forecasting_mask.float()).sum() / num_forecast_elements

            # Total Loss
            total_loss = loss_forecasting
            total_loss.backward()
            if args.clip_grad_norm is not None and args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

            epoch_train_loss += total_loss.item()
            epoch_forecast_loss += loss_forecasting.item()
            num_train_batches += 1

        avg_train_loss = epoch_train_loss / num_train_batches if num_train_batches > 0 else float("inf")
        avg_forecast_loss = epoch_forecast_loss / num_train_batches if num_train_batches > 0 else float("inf")

        # Validation
        model.eval()
        epoch_val_loss, epoch_val_forecast_loss = 0.0, 0.0
        num_val_batches = 0

        # min_val, max_val = None, None
        # if args.encoding_type == "sas":

        with torch.no_grad():
            if val_loader is not None:
                for batch_data in val_loader:
                    if batch_data is None:
                        continue

                    input_seqs = batch_data["input_sequences"].to(DEVICE)
                    goal_states = batch_data["goal_states"].to(DEVICE)
                    target_seqs = batch_data["target_sequences"].to(DEVICE)
                    lengths = batch_data["lengths"]

                    # if args.encoding_type == "sas":
                    #     input_seqs.clamp_(min=0, max=args.num_blocks)
                    #     goal_states.clamp_(min=0, max=args.num_blocks)

                    if args.encoding_type == "sas":
                        if model.domain == "blocksworld":
                            max_val = model.num_blocks
                            min_val = -1
                        elif model.domain == "grippers":
                            max_val = model.num_rooms
                            min_val = -(2 * model.num_robots)
                        else:
                            raise ValueError(f"Unsupported domain {model.domain} for SAS+ clamping.")

                        # print(f"DEBUG: Val batch SAS+ clamping to [{min_val}, {max_val}]")
                        input_seqs.clamp_(min=min_val, max=max_val)
                        goal_states.clamp_(min=min_val, max=max_val)

                    forecasting_logits, _ = model(input_seqs, goal_states, lengths)

                    # Create a mask for valid forecasting steps
                    # (B, S_max, N_blocks, N_locs)
                    if args.encoding_type == "sas":
                        mask = (
                            torch.arange(max(lengths), device=DEVICE)[None, :]
                            < lengths.clone().detach().to(DEVICE)[:, None]
                        )
                        active_logits = forecasting_logits[mask]
                        active_targets = target_seqs[mask]

                        if active_targets.numel() == 0:
                            continue

                        # Map active targets to their indices
                        active_targets_indices = model._map_sas_to_indices(active_targets)
                        loss_forecasting = criterion(
                            active_logits.reshape(-1, model.num_locations), active_targets_indices.reshape(-1)
                        )

                    else:  # Binary
                        # (B, S_max, N_blocks, N_locs)
                        forecasting_mask = torch.zeros_like(target_seqs, dtype=torch.bool).to(DEVICE)
                        for i, length_val in enumerate(lengths):
                            if length_val > 0:
                                forecasting_mask[i, :length_val, :] = True

                        # Count the number of valid forecasting elements
                        num_forecast_elements = forecasting_mask.float().sum()
                        if num_forecast_elements == 0:
                            continue

                        # Compute the loss for the valid forecasting elements
                        loss_forecasting_unreduced = criterion(forecasting_logits, target_seqs)
                        loss_forecasting = (
                            loss_forecasting_unreduced * forecasting_mask.float()
                        ).sum() / num_forecast_elements

                    total_loss = loss_forecasting
                    epoch_val_loss += total_loss.item()
                    epoch_val_forecast_loss += loss_forecasting.item()
                    num_val_batches += 1

        avg_val_loss = epoch_val_loss / num_val_batches if num_val_batches > 0 else float("inf")
        avg_val_forecast_loss = epoch_val_forecast_loss / num_val_batches if num_val_batches > 0 else float("inf")
        scheduler.step(avg_val_loss)

        # Store losses for plotting
        train_losses_history.append(avg_train_loss)
        val_losses_history.append(avg_val_loss)
        train_forecast_losses_history.append(avg_forecast_loss)
        val_forecast_losses_history.append(avg_val_forecast_loss)

        train_loss_str = f"Train Loss: {avg_train_loss:.4f} (F: {avg_forecast_loss:.4f})"
        val_loss_str = f"Val Loss: {avg_val_loss:.4f} (F: {avg_val_forecast_loss:.4f})"
        print(f"Epoch [{epoch + 1}/{args.epochs}] {train_loss_str}, {val_loss_str}")

        # Save best model
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
                    "domain": args.domain,
                    "domain_config": domain_config,
                    "embedding_dim": args.lstm_embedding_dim if args.encoding_type == "sas" else None,
                },
                str(model_save_path),
            )
            print(f"Model saved to {model_save_path} (Val Loss: {best_val_loss:.4f})")

    print("LSTM Training finished.")

    # Call plotting function after training
    plot_training_curves(
        train_losses_history,
        val_losses_history,
        train_forecast_losses_history,
        val_forecast_losses_history,
        args.epochs,
        args.model_type,
        args.encoding_type,
        config_name,
        model_save_path.parent,  # Pass the directory where the model is saved
    )


def lstm_collate_fn(batch):
    """
    Custom collate function for LSTM training. Handles padding.
    The collate function is used in DataLoader to merge a list of samples to form a mini-batch of Tensor(s).
    It is needed because each sample may have different lengths, and we need to pad them to the same length for batch processing.

    :param batch: A list of samples from the dataset. Each sample is a dict with keys:
        - 'initial_state': np.float32 array of shape (state_dim,)
        - 'goal_state': np.float32 array of shape (state_dim,)
        - 'expert_trajectory': np.float32 array of shape (L, state_dim)
        - 'id': str
    :type batch: List[Dict[str, Any]]
    :returns: A dict with the following keys:
        - 'input_sequences': Padded input sequences (B, S_max, F)
        - 'goal_states': Goal states (B, F)
        - 'target_sequences': Padded target sequences (B, S_max, F)
        - 'lengths': Original sequence lengths (B,)
        - 'ids': List of problem IDs (B,)
        - 'expert_trajectories': List of original expert trajectories as tensors (not padded) (B,)
    :rtype: Dict[str, Any]
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

    return {
        "input_sequences": padded_input_seqs,
        "goal_states": goal_states_batch,
        "target_sequences": padded_target_seqs,
        "lengths": lengths,
        "ids": ids_list,
        "expert_trajectories": expert_trajectories_orig_list,
    }
