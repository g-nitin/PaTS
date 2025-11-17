from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.data import Batch
from torch_geometric.nn import MessagePassing, global_add_pool


# Adapted from wlplan tutorial
class LinearConv(MessagePassing):
    def __init__(self, in_feat: int, out_feat: int, aggr: str = "max"):
        super().__init__(aggr=aggr)
        self.f = Linear(in_feat, out_feat, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.f(x)
        x = self.propagate(edge_index=edge_index, x=x, size=None)
        return x


class RGNNLayer(nn.Module):
    def __init__(self, n_relations: int, in_feat: int, out_feat: int):
        super(RGNNLayer, self).__init__()
        self.convs = torch.nn.ModuleList([LinearConv(in_feat, out_feat) for _ in range(n_relations)])
        self.root = Linear(in_feat, out_feat, bias=True)

    def forward(self, x: torch.Tensor, edge_indices_list: List[torch.Tensor]) -> torch.Tensor:
        x_out = self.root(x)
        for i, conv in enumerate(self.convs):
            if edge_indices_list[i].numel() > 0:  # Only apply if edges of this type exist
                x_out += conv(x, edge_indices_list[i])
        return x_out


class GNNEncoder(nn.Module):
    def __init__(self, n_relations: int, in_feat: int, embedding_dim: int, n_layers: int = 4):
        super().__init__()
        self.emb = Linear(in_feat, embedding_dim)
        self.layers = nn.ModuleList([RGNNLayer(n_relations, embedding_dim, embedding_dim) for _ in range(n_layers)])

    def forward(self, data) -> torch.Tensor:
        x, edge_indices_list, batch = data.x, data.edge_index_list, data.batch
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x, edge_indices_list)
            x = F.relu(x)

        # Global add pooling to get a graph-level embedding
        return global_add_pool(x, batch)


class Decoder(nn.Module):
    def __init__(self, embedding_dim: int, out_features: int, hidden_dim: int = 128):
        super().__init__()
        self.mlp = Sequential(Linear(embedding_dim, hidden_dim), ReLU(), Linear(hidden_dim, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class PaTS_GNN(nn.Module):
    def __init__(
        self,
        gnn_in_features: int,
        gnn_n_relations: int,
        one_hot_state_dim: int,
        gnn_embedding_dim: int = 64,
        lstm_hidden_size: int = 128,
        num_lstm_layers: int = 2,
    ):
        super().__init__()
        self.gnn_embedding_dim = gnn_embedding_dim

        self.encoder = GNNEncoder(n_relations=gnn_n_relations, in_feat=gnn_in_features, embedding_dim=gnn_embedding_dim)
        self.decoder = Decoder(embedding_dim=gnn_embedding_dim, out_features=one_hot_state_dim)
        self.lstm = nn.LSTM(
            input_size=gnn_embedding_dim * 2,  # current_embedding + goal_embedding
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
        )
        self.forecasting_head = nn.Linear(lstm_hidden_size, gnn_embedding_dim)

    def forward(self, states_batch, goal_batch, one_hot_trajectories) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encode all state graphs and goal graphs in the batch
        state_embeddings = self.encoder(states_batch)  # (total_num_states_in_batch, gnn_embedding_dim)
        goal_embeddings = self.encoder(goal_batch)  # (batch_size, gnn_embedding_dim)

        # Reconstruct sequences of embeddings from the flattened batch
        batch_size = goal_batch.num_graphs
        lengths = [len(traj) for traj in one_hot_trajectories]
        max_len = max(lengths)

        padded_embeddings = torch.zeros(batch_size, max_len, self.gnn_embedding_dim, device=state_embeddings.device)
        current_offset = 0
        for i, length in enumerate(lengths):
            padded_embeddings[i, :length] = state_embeddings[current_offset : current_offset + length]
            current_offset += length

        # Prepare for LSTM: input is S_0 to S_{L-2}, target is S_1 to S_{L-1}
        input_embeddings = padded_embeddings[:, :-1, :]
        target_embeddings_for_loss = padded_embeddings[:, 1:, :]

        # Create a mask for valid (non-padded) steps
        mask = (
            torch.arange(max_len - 1, device=input_embeddings.device)[None, :]
            < torch.tensor(lengths, device=input_embeddings.device)[:, None] - 1
        )

        # Concatenate with goal embedding for LSTM input
        goal_expanded = goal_embeddings.unsqueeze(1).repeat(1, max_len - 1, 1)
        lstm_input = torch.cat([input_embeddings, goal_expanded], dim=2)

        # LSTM forward pass
        lstm_out, _ = self.lstm(lstm_input)

        # Predict next state embeddings only for valid steps
        predicted_embeddings = self.forecasting_head(lstm_out[mask])

        # Decode predicted embeddings to reconstruct one-hot states
        reconstructed_state_logits = self.decoder(predicted_embeddings)

        # The target for the latent loss are the true embeddings of the next states
        target_embeddings_flat = target_embeddings_for_loss[mask]

        return predicted_embeddings, target_embeddings_flat, reconstructed_state_logits


def train_gnn_model_loop(model, train_loader, val_loader, args, one_hot_state_dim, config_name, model_save_path, DEVICE):
    print("\nStarting GNN training loop...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=10, factor=0.5)

    latent_criterion = nn.MSELoss()
    recon_criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            # The dataloader gives us a list of trajectories. We create a batch for the GNN encoder.
            all_graphs = [graph for traj in batch["expert_graphs"] for graph in traj]
            goal_graphs = [traj["expert_graphs"][-1] for traj in batch]  # Use final state as goal graph

            states_batch = Batch.from_data_list(all_graphs).to(DEVICE)
            goal_batch = Batch.from_data_list(goal_graphs).to(DEVICE)

            # Ground truth for reconstruction loss
            one_hot_trajectories = [item["one_hot_trajectory"] for item in batch]
            target_one_hots = torch.cat([traj[1:] for traj in one_hot_trajectories], dim=0).to(DEVICE)

            predicted_embeddings, target_embeddings, reconstructed_logits = model(
                states_batch, goal_batch, one_hot_trajectories
            )

            # 1. Latent Loss
            loss_latent = latent_criterion(predicted_embeddings, target_embeddings)

            # 2. Reconstruction Loss
            loss_recon = recon_criterion(reconstructed_logits, target_one_hots)

            # Weighted Total Loss
            total_loss = (args.latent_loss_weight * loss_latent) + (args.reconstruction_loss_weight * loss_recon)

            total_loss.backward()
            optimizer.step()
            total_train_loss += total_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                all_graphs = [graph for traj in batch["expert_graphs"] for graph in traj]
                goal_graphs = [traj["expert_graphs"][-1] for traj in batch]

                states_batch = Batch.from_data_list(all_graphs).to(DEVICE)
                goal_batch = Batch.from_data_list(goal_graphs).to(DEVICE)

                one_hot_trajectories = [item["one_hot_trajectory"] for item in batch]
                target_one_hots = torch.cat([traj[1:] for traj in one_hot_trajectories], dim=0).to(DEVICE)

                predicted_embeddings, target_embeddings, reconstructed_logits = model(
                    states_batch, goal_batch, one_hot_trajectories
                )

                loss_latent = latent_criterion(predicted_embeddings, target_embeddings)
                loss_recon = recon_criterion(reconstructed_logits, target_one_hots)
                total_val_loss += (
                    (args.latent_loss_weight * loss_latent) + (args.reconstruction_loss_weight * loss_recon)
                ).item()

        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        print(f"Epoch [{epoch + 1}/{args.epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "gnn_in_features": model.encoder.emb.in_features,
                    "gnn_n_relations": len(model.encoder.layers[0].convs),
                    "one_hot_state_dim": model.decoder.mlp[-1].out_features,
                    "gnn_embedding_dim": args.gnn_embedding_dim,
                    "lstm_hidden_size": args.lstm_hidden_size,
                    "num_lstm_layers": args.num_lstm_layers,
                },
                str(model_save_path),
            )
            print(f"Model saved to {model_save_path} (Val Loss: {best_val_loss:.4f})")
