import argparse
import re
from pathlib import Path  # For more robust path handling
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader

from ..pats_dataset import PaTSDataset

# ** Configuration **
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ** Model Definition **
class PaTS_LSTM(nn.Module):
    def __init__(self, num_features, hidden_size, num_lstm_layers, dropout_prob=0.2):
        super(PaTS_LSTM, self).__init__()
        self.num_features = num_features  # F
        self.hidden_size = hidden_size  # H
        self.num_lstm_layers = num_lstm_layers  # L

        # Input to LSTM is concat(S_t, S_G), so 2 * num_features
        self.lstm = nn.LSTM(
            input_size=2 * num_features,
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

        # Pack padded batch
        # 'lengths' should be on CPU for pack_padded_sequence
        packed_input = pack_padded_sequence(lstm_input, lengths.cpu(), batch_first=True, enforce_sorted=False)

        if h_init is None or c_init is None:
            # LSTM default initializes to zeros if not provided, but explicit is fine
            # (num_layers * num_directions, batch, hidden_size)
            # Note: num_directions is 1 for unidirectional LSTM
            h_0 = torch.zeros(self.num_lstm_layers, batch_size, self.hidden_size).to(DEVICE)
            c_0 = torch.zeros(self.num_lstm_layers, batch_size, self.hidden_size).to(DEVICE)
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


def get_num_blocks_from_filename(filename_base: str) -> Optional[int]:
    """Extracts number of blocks from a filename like 'blocks_3_problem_1'."""
    match = re.search(r"blocks_(\d+)_problem", filename_base)
    if match:
        return int(match.group(1))
    return None


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


# ** Training Function **
def train_model(model, dataloader, num_epochs, learning_rate, model_save_path, clip_grad_norm=1.0):
    criterion = nn.BCEWithLogitsLoss(reduction="none")  # Use 'none' to manually mask padding
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5, factor=0.5)

    best_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch_data in enumerate(dataloader):
            if batch_data is None:  # Skip if collate_fn returned None
                continue
            input_seqs = batch_data["input_sequences"].to(DEVICE)
            goal_states = batch_data["goal_states"].to(DEVICE)
            target_seqs = batch_data["target_sequences"].to(DEVICE)
            lengths = batch_data["lengths"]  # Keep on CPU for pack_padded_sequence

            optimizer.zero_grad()

            # Forward pass
            # output_logits: (B, S_max, F)
            output_logits, _ = model(input_seqs, goal_states, lengths)

            # Create a mask for the loss calculation based on actual sequence lengths
            # output_logits is (B, S_max, F), target_seqs is (B, S_max, F)
            mask = torch.zeros_like(target_seqs, dtype=torch.bool).to(DEVICE)
            for i, length in enumerate(lengths):
                if length > 0:  # Ensure length is positive
                    mask[i, :length, :] = True

            if (
                mask.float().sum() == 0
            ):  # Avoid division by zero if all sequences in batch are empty (should not happen with good data)
                print(f"Warning: Empty mask in batch {batch_idx}. Skipping loss calculation for this batch.")
                continue

            loss_unreduced = criterion(output_logits, target_seqs)
            loss = (loss_unreduced * mask.float()).sum() / mask.float().sum()  # Average over non-padded elements

            loss.backward()
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if batch_idx % 20 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(dataloader)}], Loss: {loss.item():.4f}"
                )

        if num_batches == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}] had no batches with data. Skipping epoch.")
            continue

        avg_epoch_loss = epoch_loss / num_batches
        scheduler.step(avg_epoch_loss)
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] completed. Avg Loss: {avg_epoch_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                    "num_features": model.num_features,
                    "hidden_size": model.hidden_size,
                    "num_lstm_layers": model.num_lstm_layers,
                    "target_num_blocks": getattr(model, "target_num_blocks", None),
                },
                model_save_path,
            )
            print(f"Model saved to {model_save_path} (Loss: {best_loss:.4f})")
    print("Training finished.")


# ** Inference (Plan Generation) Function **
def generate_plan_lstm(model, initial_state_np, goal_state_np, max_plan_length=50, num_features=None):
    """
    Generates a plan (sequence of states) using the trained LSTM model.
    """
    if num_features is None:
        num_features = model.num_features

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        current_S_tensor = torch.FloatTensor(initial_state_np).unsqueeze(0).to(DEVICE)  # (1, F)
        goal_S_tensor = torch.FloatTensor(goal_state_np).unsqueeze(0).to(DEVICE)  # (1, F)

        # Initialize hidden and cell states for the LSTM
        # (num_layers * num_directions, batch_size=1, hidden_size)
        h_prev = torch.zeros(model.num_lstm_layers, 1, model.hidden_size).to(DEVICE)
        c_prev = torch.zeros(model.num_lstm_layers, 1, model.hidden_size).to(DEVICE)

        generated_plan_states_tensors = [current_S_tensor.clone()]

        for step in range(max_plan_length):
            next_S_binary, _, h_next, c_next = model.predict_step(current_S_tensor, goal_S_tensor, h_prev, c_prev)

            generated_plan_states_tensors.append(next_S_binary.clone())
            current_S_tensor = next_S_binary
            h_prev, c_prev = h_next, c_next

            # Check if goal is reached
            if torch.equal(current_S_tensor, goal_S_tensor):
                print(f"Goal reached at step {step + 1}.")
                break

        # Convert list of tensors to a single numpy array (L, F)
        generated_plan_np = torch.cat(generated_plan_states_tensors, dim=0).cpu().numpy()
    return generated_plan_np


# ** Helper to load problem basenames from split files **
def load_problem_basenames(split_file_path: Path):
    if not split_file_path.exists():
        print(f"Warning: Split file not found: {split_file_path}")
        return []
    with open(split_file_path, "r") as f:
        basenames = [line.strip() for line in f if line.strip()]
    return basenames


# ** Main Execution **
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM Training")
    parser.add_argument("dataset_dir", type=str, help="Path to the dataset directory")
    parser.add_argument("dataset_split", type=str, help="Path to the dataset split files")
    parser.add_argument("output_dir", type=str, help="Path to the output directory")
    parser.add_argument("--num_blocks", type=int, required=True, help="Number of blocks for this training run.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=250, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    args = parser.parse_args()

    # ** Paths **
    # Base directory for the original PDDL, plans, trajectories etc.
    PATS_DATASET_DIR = Path(args.dataset_dir)
    # Directory containing train_files.txt, val_files.txt, test_files.txt
    DATA_ANALYZED_DIR = Path(args.dataset_split)
    # Directory to save models and results for this LSTM script
    LSTM_OUTPUT_DIR = Path(args.output_dir)
    LSTM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # TARGET_NUM_BLOCKS is taken from CLI if provided, or inferred if possible.
    # For simplicity, this script will rely on dataset_dir being for a specific N.
    # The num_blocks argument to the script is crucial.
    cli_num_blocks = getattr(args, "num_blocks", None)  # Check if num_blocks is passed
    if cli_num_blocks is None:
        print("Error: --num_blocks argument is required for LSTM training.")
        exit(1)
    TARGET_NUM_BLOCKS = cli_num_blocks
    print(f"**** Script configured for TARGET_NUM_BLOCKS = {TARGET_NUM_BLOCKS} ****")

    # Parameters
    HIDDEN_SIZE = 128
    NUM_LSTM_LAYERS = 2
    DROPOUT_PROB = 0.2
    LEARNING_RATE = getattr(args, "lr", 0.001)
    NUM_EPOCHS = getattr(args, "epochs", 250)
    BATCH_SIZE = getattr(args, "batch_size", 32)
    # NUM_FEATURES_FOR_N_BLOCKS will be inferred by PaTSDataset

    MODEL_SAVE_PATH = LSTM_OUTPUT_DIR / f"pats_lstm_model_N{TARGET_NUM_BLOCKS}.pth"
    RESULTS_SAVE_PATH = LSTM_OUTPUT_DIR / f"test_results_N{TARGET_NUM_BLOCKS}.json"

    # ** 1. Data Preparation using split files **
    # PaTSDataset takes dataset_dir (e.g. data/blocks_4) and split_file_name (e.g. train_files.txt)
    try:
        train_dataset = PaTSDataset(dataset_dir=PATS_DATASET_DIR, split_file_name="train_files.txt")
        val_dataset = PaTSDataset(dataset_dir=PATS_DATASET_DIR, split_file_name="val_files.txt")
        # Test dataset can be loaded if needed for an evaluation step within this script
        # test_dataset = PaTSDataset(dataset_dir=PATS_DATASET_DIR, split_file_name="test_files.txt")
    except Exception as e:
        print(f"Error initializing PaTSDataset: {e}")
        exit(1)

    if train_dataset.state_dim is None or train_dataset.state_dim <= 0:  # state_dim is inferred by PaTSDataset
        print(
            f"Could not determine num_features for {TARGET_NUM_BLOCKS} blocks from training data. "
            "Ensure train_files.txt contains valid problems for this N and files exist."
        )
        exit()

    ACTUAL_NUM_FEATURES = train_dataset.state_dim
    print(f"Using ACTUAL_NUM_FEATURES = {ACTUAL_NUM_FEATURES} (inferred) for N={TARGET_NUM_BLOCKS}")

    if len(train_dataset) == 0:
        print(f"No training data loaded for N={TARGET_NUM_BLOCKS}. Check dataset paths and split files. Exiting.")
        exit()

    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lstm_collate_fn, num_workers=0
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lstm_collate_fn, num_workers=0
    )
    # test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lstm_collate_fn, num_workers=0)

    # ** 2. Model Initialization **
    model = PaTS_LSTM(ACTUAL_NUM_FEATURES, HIDDEN_SIZE, NUM_LSTM_LAYERS, DROPOUT_PROB).to(DEVICE)
    setattr(model, "target_num_blocks", TARGET_NUM_BLOCKS)  # Store N in model for reference
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    # ** 3. Training **
    CLIP_GRAD_NORM = 1.0  # Gradient clipping norm, can be None to disable
    print(f"Starting training for N={TARGET_NUM_BLOCKS}...")
    train_model(model, train_dataloader, NUM_EPOCHS, LEARNING_RATE, MODEL_SAVE_PATH, CLIP_GRAD_NORM)

    if Path(MODEL_SAVE_PATH).exists():
        print(f"LSTM Model training/loading complete. Model available at {MODEL_SAVE_PATH}")
        print("To evaluate, use the benchmark.py script.")
    else:
        print("LSTM Model training was skipped and no pre-trained model found.")

    print(f"**** LSTM Script for TARGET_NUM_BLOCKS = {TARGET_NUM_BLOCKS} finished. ****")
