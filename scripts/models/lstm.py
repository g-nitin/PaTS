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
