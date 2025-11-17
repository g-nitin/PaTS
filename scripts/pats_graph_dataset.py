import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from wlplan.graph_generator import init_graph_generator
from wlplan.planning import Atom, State, parse_domain


class PaTSGraphDataset(Dataset):
    """
    A PyTorch Dataset for loading PaTS data as sequences of graphs and their
    corresponding one-hot vector representations.
    """

    def __init__(
        self,
        raw_data_dir: Path,
        processed_data_dir: Path,
        pddl_domain_file: Path,
        split_file_name: str,
    ):
        """
        Initializes the PaTSGraphDataset.

        :param raw_data_dir: The root directory for raw problem data for a specific config.
        :param processed_data_dir: The directory for processed one-hot encoded trajectories.
        :param pddl_domain_file: Path to the PDDL domain file.
        :param split_file_name: The name of the file containing problem basenames for this split.
        """
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.pddl_domain_file = pddl_domain_file
        self.split_file_path = self.raw_data_dir / "splits" / split_file_name
        self.trajectories_text_dir = self.raw_data_dir / "trajectories_text"

        self.basenames = self._load_basenames()
        if not self.basenames:
            raise ValueError(f"No basenames loaded from {self.split_file_path}.")

        # Initialize wlplan domain and graph generator
        self.wlplan_domain = parse_domain(str(self.pddl_domain_file))
        self.graph_generator = init_graph_generator(graph_representation="ilg", domain=self.wlplan_domain)
        self.name_to_predicate = {p.name: p for p in self.wlplan_domain.predicates}

    def _load_basenames(self) -> List[str]:
        if not self.split_file_path.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_file_path}")
        with open(self.split_file_path, "r") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]

    def _text_state_to_wlplan_state(self, text_state: str) -> State:
        """Converts a single line of text predicates into a wlplan.planning.State."""
        atoms = []
        # Regex to handle predicates like (on b1 b2)
        predicate_matches = re.findall(r"\((.*?)\)", text_state)
        for match in predicate_matches:
            parts = match.strip().split()
            pred_name = parts[0]
            if pred_name in self.name_to_predicate:
                predicate = self.name_to_predicate[pred_name]
                args = parts[1:]
                if len(args) == len(predicate.parameters):
                    atoms.append(Atom(predicate=predicate, objects=args))
        return State(atoms)

    def _wlplan_graph_to_pyg(self, graph: "wlplan.graph_generator.Graph") -> Data:
        """Converts a wlplan.Graph to a torch_geometric.data.Data object."""
        nodes = graph.node_colours
        edges = graph.edges

        # Node features: one-hot encoding of node colours
        x = torch.zeros(len(nodes), self.graph_generator.get_n_features())
        x[torch.arange(len(nodes)), nodes] = 1

        edge_index_list = [[] for _ in range(self.graph_generator.get_n_relations())]
        for u, neighbours in enumerate(edges):
            for r, v in neighbours:
                edge_index_list[r].append([u, v])

        edge_index_tensors = [torch.tensor(e, dtype=torch.long).t().contiguous() for e in edge_index_list]

        return Data(x=x, edge_index_list=edge_index_tensors)

    def __len__(self) -> int:
        return len(self.basenames)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        basename = self.basenames[idx]
        traj_text_path = self.trajectories_text_dir / f"{basename}.traj.txt"
        traj_one_hot_path = self.processed_data_dir / f"{basename}.traj.bin.npy"  # Path to one-hot file

        if not traj_text_path.exists():
            raise FileNotFoundError(f"Trajectory text file not found: {traj_text_path}")
        if not traj_one_hot_path.exists():
            raise FileNotFoundError(f"Trajectory one-hot file not found: {traj_one_hot_path}")

        with open(traj_text_path, "r") as f:
            lines = [line.strip() for line in f if line.strip() and "Goal Predicates:" not in line]

        # Load one-hot vectors
        one_hot_trajectory = torch.from_numpy(np.load(traj_one_hot_path)).float()

        # Ensure consistency
        if len(lines) != one_hot_trajectory.shape[0]:
            # This can happen if the text file has an extra newline, etc.
            # We trust the .npy file as the source of truth for length.
            lines = lines[: one_hot_trajectory.shape[0]]

        expert_graphs = []
        for line in lines:
            wl_state = self._text_state_to_wlplan_state(line)
            # Note: Problem context is not set here, which is okay for ILG representation
            wl_graph = self.graph_generator.to_graph(wl_state)
            pyg_data = self._wlplan_graph_to_pyg(wl_graph)
            expert_graphs.append(pyg_data)

        return {
            "expert_graphs": expert_graphs,
            "one_hot_trajectory": one_hot_trajectory,
            "id": basename,
        }
