import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

# ** Configuration **
DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))


# ** Model Definition **
class PaTS_LSTM(nn.Module):
    def __init__(self, num_features, hidden_size, num_lstm_layers, dropout_prob=0.2):
        super(PaTS_LSTM, self).__init__()
        self.num_features = num_features  # F
        self.hidden_size = hidden_size  # H
        self.num_lstm_layers = num_lstm_layers  # L

        # Input to LSTM is concat(S_t, S_G), so 2 * num_features
        self.lstm = nn.LSTM(
            input_size=2 * num_features,  # Input to LSTM is concat(S_t, S_G)
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,  # Expects (batch, seq_len, features)
            dropout=dropout_prob if num_lstm_layers > 1 else 0.0,
        )

        self.fc_out = nn.Linear(hidden_size, num_features)
        # BCEWithLogitsLoss will handle the sigmoid for training

    def forward(self, current_states_batch, goal_state_batch, lengths, h_init=None, c_init=None):
        """
        Forward pass for training or multi-step inference.
        Args:
            current_states_batch (Tensor): Batch of current state sequences (B, S_max, F). Padded sequences.
            goal_state_batch (Tensor): Batch of goal states (B, F).
            lengths (Tensor): Batch of original sequence lengths (B,). For packing.
            h_init (Tensor, optional): Initial hidden state.
            c_init (Tensor, optional): Initial cell state.
        Returns:
            output_logits (Tensor): Logits for predicted next states (B, S_max, F).
            (h_n, c_n) (tuple): Last hidden and cell states.
        """
        batch_size, max_seq_len, _ = current_states_batch.shape

        # Expand goal state to match sequence length for concatenation
        # goal_state_expanded: (B, S_max, F)
        goal_state_expanded = goal_state_batch.unsqueeze(1).repeat(1, max_seq_len, 1)

        # Concatenate current state sequences and goal states along the feature dimension
        # lstm_input: (B, S_max, 2 * F)
        lstm_input = torch.cat((current_states_batch, goal_state_expanded), dim=2)

        # lengths should be on CPU for pack_padded_sequence
        packed_input = pack_padded_sequence(lstm_input, lengths.cpu(), batch_first=True, enforce_sorted=False)

        if h_init is None or c_init is None:
            h_0 = torch.zeros(self.num_lstm_layers, batch_size, self.hidden_size).to(
                current_states_batch.device
            )  # Use input device
            c_0 = torch.zeros(self.num_lstm_layers, batch_size, self.hidden_size).to(
                current_states_batch.device
            )  # Use input device
        else:
            h_0, c_0 = h_init, c_init

        packed_output, (h_n, c_n) = self.lstm(packed_input, (h_0, c_0))

        # Unpack sequence
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=max_seq_len)
        # lstm_out: (B, S_max, H)

        # Pass LSTM output through the fully connected layer
        # output_logits: (B, S_max, F)
        output_logits = self.fc_out(lstm_out)

        return output_logits, (h_n, c_n)

    def predict_step(self, current_state_S_t, goal_state_S_G, h_prev, c_prev):
        """
        Predicts the single next state for inference.
        Args:
            current_state_S_t (Tensor): Current state (1, F).
            goal_state_S_G (Tensor): Goal state (1, F).
            h_prev (Tensor): Previous hidden state from LSTM.
            c_prev (Tensor): Previous cell state from LSTM.
        Returns:
            next_state_binary (Tensor): Predicted binary next state (1, F).
            next_state_probs (Tensor): Predicted probabilities for next state (1, F).
            (h_next, c_next) (tuple): New hidden and cell states.
        """
        self.eval()  # Set to evaluation mode

        # Reshape S_t to (batch_size=1, seq_len=1, num_features) for LSTM input
        current_state_S_t_seq = current_state_S_t.unsqueeze(1)  # (1, 1, F)
        goal_state_S_G_expanded = goal_state_S_G.unsqueeze(1)  # (1, 1, F)

        lstm_input_step = torch.cat((current_state_S_t_seq, goal_state_S_G_expanded), dim=2)  # (1, 1, 2F)

        # LSTM expects (h_0, c_0) even for a single step if states are passed
        lstm_out, (h_next, c_next) = self.lstm(lstm_input_step, (h_prev, c_prev))
        # lstm_out: (1, 1, H)

        next_state_logits = self.fc_out(lstm_out.squeeze(1))  # (1, H) -> (1, F)
        next_state_probs = torch.sigmoid(next_state_logits)

        # Convert probabilities to binary state (e.g., thresholding)
        next_state_binary = (next_state_probs > 0.5).float()

        return next_state_binary, next_state_probs, h_next, c_next


def lstm_collate_fn(batch):
    # Filter out None items that might result from __getitem__ errors
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    input_seqs_list = []
    target_seqs_list = []
    goal_states_list = []
    expert_trajectories_orig_list = []  # Store original expert trajectories
    ids_list = []

    for item in batch:
        # item is a dict from PaTSDataset: {'initial_state', 'goal_state', 'expert_trajectory', 'id'}
        # All values are np.float32 arrays.
        expert_trajectory_np = item["expert_trajectory"]

        if expert_trajectory_np.shape[0] == 1:  # S0 is goal or only S0 exists
            input_s_np = expert_trajectory_np  # S0
            target_s_np = expert_trajectory_np  # S0 (predict S0 from S0)
        else:  # expert_trajectory_np.shape[0] > 1
            input_s_np = expert_trajectory_np[:-1, :]  # S_0 to S_{T-1}
            target_s_np = expert_trajectory_np[1:, :]  # S_1 to S_T

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
        "expert_trajectories": expert_trajectories_orig_list,  # List of original trajectory tensors
    }
