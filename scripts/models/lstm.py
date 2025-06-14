import argparse
from pathlib import Path  # For more robust path handling

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from helpers import get_num_blocks_from_filename
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

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


class BWTrajectoryDataset(Dataset):
    def __init__(self, problem_basenames, pats_dataset_dir: Path, target_num_blocks: int):
        self.problem_basenames = []
        self.trajectory_files = []
        self.goal_files = []
        self.num_features = None
        self.target_num_blocks = target_num_blocks

        print(f"Initializing dataset for {target_num_blocks} blocks.")
        processed_count = 0
        skipped_count = 0

        for basename in problem_basenames:
            num_blocks_in_file = get_num_blocks_from_filename(basename)
            if num_blocks_in_file != self.target_num_blocks:
                skipped_count += 1
                continue  # Skip files not matching the target number of blocks

            traj_path = pats_dataset_dir / "trajectories_bin" / f"{basename}.traj.bin.npy"
            goal_path = pats_dataset_dir / "trajectories_bin" / f"{basename}.goal.bin.npy"

            if not traj_path.exists() or not goal_path.exists():
                # print(f"Warning: Data files not found for {basename}. Skipping.")
                skipped_count += 1
                continue

            # Determine num_features from the first valid file
            if self.num_features is None:
                try:
                    temp_traj_data = np.load(traj_path)
                    if temp_traj_data.ndim == 2 and temp_traj_data.shape[0] > 0:
                        self.num_features = temp_traj_data.shape[1]
                        print(
                            f"Determined num_features = {self.num_features} from {traj_path} for {self.target_num_blocks} blocks."
                        )
                    else:
                        print(f"Warning: Trajectory file {traj_path} is empty or malformed. Skipping {basename}.")
                        skipped_count += 1
                        continue
                except Exception as e:
                    print(f"Error loading {traj_path} to determine num_features: {e}. Skipping {basename}.")
                    skipped_count += 1
                    continue

            self.problem_basenames.append(basename)
            self.trajectory_files.append(traj_path)
            self.goal_files.append(goal_path)
            processed_count += 1

        print(
            f"Dataset initialized: Loaded {processed_count} problems for {target_num_blocks} blocks. Skipped {skipped_count} problems."
        )
        if processed_count == 0 and len(problem_basenames) > 0:
            print(f"Warning: No problems loaded for {target_num_blocks} blocks. Check paths and file names.")

    def __len__(self):
        return len(self.trajectory_files)

    def __getitem__(self, idx):
        traj_path = self.trajectory_files[idx]
        goal_path = self.goal_files[idx]
        basename = self.problem_basenames[idx]

        try:
            trajectory_np = np.load(traj_path)
            goal_np = np.load(goal_path)
        except Exception as e:
            print(f"Error loading .npy files for {basename}: {e}. Returning None or dummy.")
            # Handle error, e.g., by returning a dummy item or raising an error
            # For now, let's try to skip by getting next item (simplistic)
            return self.__getitem__((idx + 1) % len(self)) if len(self) > 0 else None

        if trajectory_np.shape[0] < 1:  # Must have at least S0
            print(f"Warning: Trajectory {traj_path} is empty. Skipping.")
            return self.__getitem__((idx + 1) % len(self)) if len(self) > 0 else None

        if trajectory_np.shape[0] == 1:  # S0 is goal
            input_seq_np = trajectory_np  # S0
            target_seq_np = trajectory_np  # S0
        else:  # trajectory_np.shape[0] > 1
            input_seq_np = trajectory_np[:-1, :]  # S_0 to S_{T-1}
            target_seq_np = trajectory_np[1:, :]  # S_1 to S_T

        return {
            "input_sequence": torch.FloatTensor(input_seq_np),
            "goal_state": torch.FloatTensor(goal_np),
            "target_sequence": torch.FloatTensor(target_seq_np),
            "expert_trajectory": torch.FloatTensor(trajectory_np),  # Full S_0 to S_T
            "id": basename,
        }


def collate_fn(batch):
    # Filter out None items that might result from __getitem__ errors
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    input_seqs = [item["input_sequence"] for item in batch]
    goal_states = torch.stack([item["goal_state"] for item in batch])
    target_seqs = [item["target_sequence"] for item in batch]
    expert_trajectories = [item["expert_trajectory"] for item in batch]  # For evaluation
    ids = [item["id"] for item in batch]

    lengths = torch.tensor([len(seq) for seq in input_seqs], dtype=torch.long)

    # Pad sequences
    # pad_sequence expects a list of tensors (seq_len, features)
    # and returns (max_seq_len, batch_size, features) if batch_first=False (default)
    # or (batch_size, max_seq_len, features) if batch_first=True
    padded_input_seqs = pad_sequence(input_seqs, batch_first=True, padding_value=0.0)
    padded_target_seqs = pad_sequence(target_seqs, batch_first=True, padding_value=0.0)
    # Expert trajectories are not directly fed to LSTM in this form, so padding them for convenience
    # For evaluation, we typically process them one by one.

    return {
        "input_sequences": padded_input_seqs,
        "goal_states": goal_states,
        "target_sequences": padded_target_seqs,
        "lengths": lengths,
        "ids": ids,
        "expert_trajectories": expert_trajectories,  # List of tensors
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
    args = parser.parse_args()

    # ** Paths **
    # Base directory for the original PDDL, plans, trajectories etc.
    PATS_DATASET_DIR = Path(args.dataset_dir)
    # Directory containing train_files.txt, val_files.txt, test_files.txt
    DATA_ANALYZED_DIR = Path(args.dataset_split)
    # Directory to save models and results for this LSTM script
    LSTM_OUTPUT_DIR = Path(args.output_dir)
    LSTM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Parameters
    # The script will filter data from split files for this specific N.
    # A separate model will be trained for each N.
    TARGET_NUM_BLOCKS = 4
    NUM_FEATURES_FOR_N_BLOCKS = 25
    HIDDEN_SIZE = 128
    NUM_LSTM_LAYERS = 2
    DROPOUT_PROB = 0.2
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 250
    BATCH_SIZE = 32

    print(f"**** Script configured for TARGET_NUM_BLOCKS = {TARGET_NUM_BLOCKS} ****")

    MODEL_SAVE_PATH = LSTM_OUTPUT_DIR / f"pats_lstm_model_N{TARGET_NUM_BLOCKS}.pth"
    RESULTS_SAVE_PATH = LSTM_OUTPUT_DIR / f"test_results_N{TARGET_NUM_BLOCKS}.json"

    # ** 1. Data Preparation using split files **
    train_basenames_all = load_problem_basenames(DATA_ANALYZED_DIR / "train_files.txt")
    val_basenames_all = load_problem_basenames(DATA_ANALYZED_DIR / "val_files.txt")
    test_basenames_all = load_problem_basenames(DATA_ANALYZED_DIR / "test_files.txt")

    if not train_basenames_all:
        print("No training files loaded. Exiting.")
        exit()

    # Create datasets filtered for TARGET_NUM_BLOCKS
    # The dataset class itself will handle filtering and determine num_features
    train_dataset = BWTrajectoryDataset(train_basenames_all, PATS_DATASET_DIR, TARGET_NUM_BLOCKS)

    # Ensure num_features was set (i.e., at least one valid problem was found)
    if train_dataset.num_features is None:
        print(
            f"Could not determine num_features for {TARGET_NUM_BLOCKS} blocks from training data. "
            "Ensure train_files.txt contains valid problems for this N and files exist."
        )
        exit()

    ACTUAL_NUM_FEATURES = train_dataset.num_features
    print(f"Using ACTUAL_NUM_FEATURES = {ACTUAL_NUM_FEATURES} for N={TARGET_NUM_BLOCKS}")

    val_dataset = BWTrajectoryDataset(val_basenames_all, PATS_DATASET_DIR, TARGET_NUM_BLOCKS)
    test_dataset = BWTrajectoryDataset(test_basenames_all, PATS_DATASET_DIR, TARGET_NUM_BLOCKS)

    if len(train_dataset) == 0:
        print(f"No training data loaded for N={TARGET_NUM_BLOCKS}. Check dataset paths and split files. Exiting.")
        exit()

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)

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
