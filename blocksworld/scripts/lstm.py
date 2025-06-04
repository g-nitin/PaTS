import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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


# ** Dataset and DataLoader **
class BWTrajectoryDataset(Dataset):
    def __init__(self, trajectory_files, goal_files):
        self.trajectory_files = trajectory_files
        self.goal_files = goal_files
        assert len(self.trajectory_files) == len(self.goal_files), "Mismatch in trajectory and goal files"

    def __len__(self):
        return len(self.trajectory_files)

    def __getitem__(self, idx):
        traj_path = self.trajectory_files[idx]
        goal_path = self.goal_files[idx]

        # Load binary encoded trajectory S_0, S_1, ..., S_T
        # Shape: (L, F) where L is trajectory length
        trajectory_np = np.load(traj_path)

        # Load binary encoded goal state S_G
        # Shape: (F,)
        goal_np = np.load(goal_path)

        # Input sequence for LSTM: S_0, ..., S_{T-1}
        # Target sequence for LSTM: S_1, ..., S_T
        # Ensure trajectory has at least 2 states (S0, S1)
        if trajectory_np.shape[0] < 2:
            # This can happen if a plan is just 1 state (initial = goal) or empty.
            # Handle by skipping or returning a dummy. For now, safe to assume valid plans.
            if trajectory_np.shape[0] == 1:
                input_seq_np = trajectory_np
                target_seq_np = trajectory_np
            else:  # Should not happen with FastDownward plans for non-trivial problems
                raise ValueError(f"Warning: Trajectory {traj_path} has < 2 states.")

        input_seq_np = trajectory_np[:-1, :]  # S_0 to S_{T-1}
        target_seq_np = trajectory_np[1:, :]  # S_1 to S_T

        return {
            "input_sequence": torch.FloatTensor(input_seq_np),
            "goal_state": torch.FloatTensor(goal_np),
            "target_sequence": torch.FloatTensor(target_seq_np),
            "id": os.path.basename(traj_path),  # For debugging
        }


def collate_fn(batch):
    """
    Pads sequences in a batch and prepares them for `pack_padded_sequence`.
    """
    input_seqs = [item["input_sequence"] for item in batch]
    goal_states = torch.stack([item["goal_state"] for item in batch])
    target_seqs = [item["target_sequence"] for item in batch]
    ids = [item["id"] for item in batch]

    # Get sequence lengths BEFORE padding
    lengths = torch.tensor([len(seq) for seq in input_seqs], dtype=torch.long)

    # Pad sequences
    # pad_sequence expects a list of tensors (seq_len, features)
    # and returns (max_seq_len, batch_size, features) if batch_first=False (default)
    # or (batch_size, max_seq_len, features) if batch_first=True
    padded_input_seqs = pad_sequence(input_seqs, batch_first=True, padding_value=0.0)
    padded_target_seqs = pad_sequence(target_seqs, batch_first=True, padding_value=0.0)  # Pad with 0s

    return {
        "input_sequences": padded_input_seqs,
        "goal_states": goal_states,
        "target_sequences": padded_target_seqs,
        "lengths": lengths,
        "ids": ids,
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
                mask[i, :length, :] = True

            # Apply loss only on non-padded parts
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

        avg_epoch_loss = epoch_loss / num_batches
        scheduler.step(avg_epoch_loss)
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] completed. Average Loss: {avg_epoch_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                    "num_features": model.num_features,  # Save for reloading
                    "hidden_size": model.hidden_size,
                    "num_lstm_layers": model.num_lstm_layers,
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

        generated_plan_states = [current_S_tensor.cpu().numpy().squeeze(0)]  # Store as numpy arrays

        for step in range(max_plan_length):
            next_S_binary, _, h_next, c_next = model.predict_step(current_S_tensor, goal_S_tensor, h_prev, c_prev)

            generated_plan_states.append(next_S_binary.cpu().numpy().squeeze(0))
            current_S_tensor = next_S_binary
            h_prev, c_prev = h_next, c_next

            # Check if goal is reached
            if torch.equal(current_S_tensor, goal_S_tensor):
                print(f"Goal reached at step {step + 1}.")
                break
            # Check for stagnation (optional, more complex)
            # if len(generated_plan_states) > 2 and np.array_equal(generated_plan_states[-1], generated_plan_states[-2]):
            #     print("Stagnation detected.")
            #     break
        else:  # Loop finished without break (max_plan_length reached)
            print(f"Max plan length ({max_plan_length}) reached.")

    return np.array(generated_plan_states)


# ** Main Execution Example (Illustrative) **
if __name__ == "__main__":
    # ** Parameters **
    # These should be determined by your dataset, specifically the number of blocks
    # For 3 blocks (A,B,C):
    # A_on_table, B_on_table, C_on_table (3)
    # A_on_B, A_on_C, B_on_A, B_on_C, C_on_A, C_on_B (6)
    # A_clear, B_clear, C_clear (3)
    # A_held, B_held, C_held (3)
    # arm-empty (1) (Assuming this is part of your encoding, if not, adjust)
    # Total for 3 blocks: 3+6+3+3+1 = 16 features (if arm-empty is included)
    # Update this based on your actual parse_and_encode.py output for a given N
    NUM_FEATURES_FOR_N_BLOCKS = 16  # Example for 3 blocks, adjust this!
    HIDDEN_SIZE = 128
    NUM_LSTM_LAYERS = 2
    DROPOUT_PROB = 0.2
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 250
    BATCH_SIZE = 32

    # Current dir
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_SAVE_PATH = os.path.join(ROOT_DIR, "lstm", "pats_lstm_model.pth")
    DATASET_BASE_DIR = "pats_dataset"

    # ** 1. Data Preparation **
    # Glob for trajectory and goal files.
    # This assumes all .npy files in trajectories_bin and goals_bin are for the *same* number of blocks
    # and thus have the same NUM_FEATURES_FOR_N_BLOCKS.
    # AKA, assume training one model per N_BLOCKS configuration.

    # Example: find all files for a specific number of blocks, e.g., 3 blocks
    # You might need to adjust the glob pattern if your naming is different or includes N directly
    # For example, if files are named 'blocks_3_problem_X.traj.bin.npy'
    num_blocks_for_training = 3  # Set this to the N you are training for
    print(f"Looking for data for {num_blocks_for_training} blocks...")

    # Adjust glob patterns based on your exact file naming from generate_dataset.sh
    # This pattern assumes files like 'blocks_3_problem_1.traj.bin.npy'
    # If your script produces 'blocks_03_...' or similar, adjust the pattern.
    # The '*' will match any problem ID.
    traj_pattern = os.path.join(
        DATASET_BASE_DIR, "trajectories_bin", f"blocks_{num_blocks_for_training}_problem_*.traj.bin.npy"
    )
    goal_pattern = os.path.join(
        DATASET_BASE_DIR, "trajectories_bin", f"blocks_{num_blocks_for_training}_problem_*.goal.bin.npy"
    )  # Goal files are also in trajectories_bin as per your description

    all_traj_files = sorted(glob.glob(traj_pattern))
    all_goal_files = sorted(glob.glob(goal_pattern))

    # Sanity check: ensure corresponding files exist
    # The goal file name should match the trajectory file name (except for .goal vs .traj)
    # Your structure says:
    # trajectories_bin/blocks_<N>_problem_<M>.traj.bin.npy
    # trajectories_bin/blocks_<N>_problem_<M>.goal.bin.npy
    # So the glob patterns above should work if the base names match.
    # Let's refine the file matching to be absolutely sure:

    valid_traj_files = []
    valid_goal_files = []
    for traj_f in all_traj_files:
        # Construct expected goal file name from trajectory file name
        base_name = os.path.basename(traj_f).replace(".traj.bin.npy", ".goal.bin.npy")
        expected_goal_f = os.path.join(os.path.dirname(traj_f), base_name)  # Goals are in the same dir
        if os.path.exists(expected_goal_f):
            valid_traj_files.append(traj_f)
            valid_goal_files.append(expected_goal_f)
        else:
            print(f"Warning: Goal file not found for {traj_f}: expected {expected_goal_f}")

    if not valid_traj_files:
        print(
            f"No trajectory/goal pairs found for {num_blocks_for_training} blocks with pattern: {traj_pattern}. Exiting."
        )
        print("Please check your DATASET_BASE_DIR, num_blocks_for_training, and file naming.")
        exit()

    print(f"Found {len(valid_traj_files)} trajectory/goal pairs for {num_blocks_for_training} blocks.")

    # Determine NUM_FEATURES_FOR_N_BLOCKS dynamically from the first file if not hardcoded
    # This is safer.
    temp_traj = np.load(valid_traj_files[0])
    ACTUAL_NUM_FEATURES = temp_traj.shape[1]
    print(f"Dynamically determined NUM_FEATURES: {ACTUAL_NUM_FEATURES}")
    if (
        NUM_FEATURES_FOR_N_BLOCKS != ACTUAL_NUM_FEATURES and NUM_FEATURES_FOR_N_BLOCKS is not None
    ):  # if it was hardcoded
        print(
            f"Warning: Hardcoded NUM_FEATURES_FOR_N_BLOCKS ({NUM_FEATURES_FOR_N_BLOCKS}) "
            f"does not match data ({ACTUAL_NUM_FEATURES}). Using data's value."
        )
    NUM_FEATURES_FOR_N_BLOCKS = ACTUAL_NUM_FEATURES

    # Split into train/validation (e.g., 80/20)
    # Ensure consistent shuffling if you run this multiple times
    np.random.seed(42)
    indices = np.arange(len(valid_traj_files))
    np.random.shuffle(indices)
    split_idx = int(len(indices) * 0.8)

    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_traj_files = [valid_traj_files[i] for i in train_indices]
    train_goal_files = [valid_goal_files[i] for i in train_indices]
    val_traj_files = [valid_traj_files[i] for i in val_indices]
    val_goal_files = [valid_goal_files[i] for i in val_indices]

    train_dataset = BWTrajectoryDataset(train_traj_files, train_goal_files)
    val_dataset = BWTrajectoryDataset(val_traj_files, val_goal_files)  # For validation later

    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2
    )
    # val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # ** 2. Model Initialization **
    model = PaTS_LSTM(NUM_FEATURES_FOR_N_BLOCKS, HIDDEN_SIZE, NUM_LSTM_LAYERS, DROPOUT_PROB).to(DEVICE)
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    # ** 3. Training **
    # Check if you want to train or load a pre-trained model
    TRAIN_NEW_MODEL = True  # Set to False to skip training and go to inference (if model exists)

    if TRAIN_NEW_MODEL:
        print("Starting training...")
        train_model(model, train_dataloader, NUM_EPOCHS, LEARNING_RATE, MODEL_SAVE_PATH)
    else:
        print(f"Skipping training. Attempting to load model from {MODEL_SAVE_PATH}")
        if os.path.exists(MODEL_SAVE_PATH):
            checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
            # Ensure model architecture matches saved model
            loaded_num_features = checkpoint.get("num_features", NUM_FEATURES_FOR_N_BLOCKS)  # Fallback for older saves
            loaded_hidden_size = checkpoint.get("hidden_size", HIDDEN_SIZE)
            loaded_num_lstm_layers = checkpoint.get("num_lstm_layers", NUM_LSTM_LAYERS)

            if (
                loaded_num_features != NUM_FEATURES_FOR_N_BLOCKS
                or loaded_hidden_size != HIDDEN_SIZE
                or loaded_num_lstm_layers != NUM_LSTM_LAYERS
            ):
                print("Warning: Model architecture in checkpoint differs from current parameters.")
                print("Re-initializing model with checkpoint parameters.")
                NUM_FEATURES_FOR_N_BLOCKS = loaded_num_features
                HIDDEN_SIZE = loaded_hidden_size
                NUM_LSTM_LAYERS = loaded_num_lstm_layers
                model = PaTS_LSTM(NUM_FEATURES_FOR_N_BLOCKS, HIDDEN_SIZE, NUM_LSTM_LAYERS, DROPOUT_PROB).to(DEVICE)

            model.load_state_dict(checkpoint["model_state_dict"])
            print(
                f"Model loaded successfully from {MODEL_SAVE_PATH}. Epoch: {checkpoint.get('epoch', 'N/A')}, Loss: {checkpoint.get('loss', 'N/A'):.4f}"
            )
        else:
            print(f"Model file not found at {MODEL_SAVE_PATH}. Please train first or check path.")
            exit()

    # ** 4. Inference Example **
    if len(val_dataset) > 0:
        print("\n** Running Inference on a validation sample **")
        sample_idx = 0  # Take the first validation sample
        sample_data = val_dataset[sample_idx]

        initial_state_np = sample_data["input_sequence"][0].numpy()  # S_0 from the trajectory
        goal_state_np = sample_data["goal_state"].numpy()
        expert_trajectory_np = np.vstack((initial_state_np, sample_data["target_sequence"].numpy()))

        print(f"Problem ID for inference: {sample_data['id']}")
        print(f"Initial State (first few features): {initial_state_np[:10]}...")
        print(f"Goal State (first few features): {goal_state_np[:10]}...")

        generated_plan = generate_plan_lstm(
            model,
            initial_state_np,
            goal_state_np,
            max_plan_length=len(expert_trajectory_np) + 10,  # Give some slack
            num_features=NUM_FEATURES_FOR_N_BLOCKS,
        )

        print(f"\nGenerated Plan (length {len(generated_plan)}):")
        # for i, state_vec in enumerate(generated_plan):
        #     print(f"Step {i}: {state_vec[:10]}...") # Print first few features

        # Basic evaluation:
        plan_is_valid_trajectory = True  # Placeholder for actual validation
        reached_goal = np.array_equal(generated_plan[-1], goal_state_np)
        plan_length = len(generated_plan) - 1  # Number of actions

        print(f"\nInference Summary for {sample_data['id']}:")
        print(f"  Expert plan length: {len(expert_trajectory_np) - 1}")
        print(f"  Generated plan length: {plan_length}")
        print(f"  Reached goal: {reached_goal}")

        # To calculate planning accuracy as in your abstract, you'd need to:
        # 1. Run inference on all problems in a test set.
        # 2. For each problem, check if the generated plan reaches the goal.
        # 3. Check if all intermediate states in the generated plan are valid (this is harder, might need VAL or a domain checker).
        # Planning Accuracy = (Number of problems where a valid plan reaching the goal was found) / (Total test problems)
    else:
        print("No validation data to run inference on.")
