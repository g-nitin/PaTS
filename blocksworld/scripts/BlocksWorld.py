import itertools
import random
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import torch


@dataclass
class BlockState:
    """Represents the state of blocks in the world following PDDL predicates"""

    clear: Set[str]  # blocks with nothing on top (clear ?x)
    on_table: Set[str]  # blocks on table (on-table ?x)
    on: Dict[str, str]  # block stacked on another block (on ?x ?y)
    holding: str | None = None  # block being held, if any (holding ?x)

    @property
    def arm_empty(self) -> bool:
        return self.holding is None

    def copy(self):
        return BlockState(
            clear=self.clear.copy(),
            on_table=self.on_table.copy(),
            holding=self.holding,
            on=self.on.copy(),
        )


class BlocksWorldGenerator:
    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.blocks = [chr(65 + i) for i in range(num_blocks)]  # A, B, C, ...
        self._initialize_feature_names()

    def _initialize_feature_names(self):
        """Initialize the names of all binary features for the encoding"""
        self.feature_names = []

        # Features for blocks on table
        self.feature_names.extend([f"{b}_on_table" for b in self.blocks])

        # Features for blocks on other blocks
        for b1, b2 in itertools.product(self.blocks, self.blocks):
            if b1 != b2:
                self.feature_names.append(f"{b1}_on_{b2}")

        # Features for clear blocks
        self.feature_names.extend([f"{b}_clear" for b in self.blocks])

        # Features for held blocks
        self.feature_names.extend([f"{b}_held" for b in self.blocks])

        # Create index mapping for quick encoding
        self.feature_to_idx = {name: idx for idx, name in enumerate(self.feature_names)}

    def generate_random_state(self) -> BlockState:
        """Generate a random valid state with arm empty"""
        clear = set(self.blocks)
        on_table = set()
        on = {}
        available = self.blocks.copy()

        while available:
            block = available.pop()
            if random.random() < 0.5 or not available:  # 50% chance or last block
                on_table.add(block)
            else:
                possible_bases = list(clear - {block})
                if possible_bases:
                    base = random.choice(possible_bases)
                    on[block] = base
                    clear.remove(base)
                else:
                    on_table.add(block)

        return BlockState(clear=clear, on_table=on_table, holding=None, on=on)

    def get_valid_actions(self, state: BlockState) -> List[Tuple[str, str, str]]:
        """Get all valid actions in current state based on PDDL preconditions"""
        actions = []

        if state.holding:
            # putdown action
            actions.append(("putdown", state.holding, None))

            # stack action - can stack on any clear block
            for underob in state.clear:
                if underob != state.holding:
                    actions.append(("stack", state.holding, underob))
        else:
            # pickup action - any clear block on table
            for ob in state.clear & state.on_table:
                actions.append(("pickup", ob, None))

            # unstack action - any clear block that's on another block
            for ob, underob in state.on.items():
                if ob in state.clear:
                    actions.append(("unstack", ob, underob))

        return actions

    def apply_action(self, state: BlockState, action: Tuple[str, str, str]) -> BlockState:
        """Apply action to state following PDDL effects"""
        new_state = state.copy()
        action_type, ob, underob = action

        if action_type == "pickup":
            assert ob in state.clear and ob in state.on_table and state.arm_empty
            new_state.holding = ob
            new_state.clear.remove(ob)
            new_state.on_table.remove(ob)

        elif action_type == "putdown":
            assert state.holding == ob
            new_state.clear.add(ob)
            new_state.on_table.add(ob)
            new_state.holding = None

        elif action_type == "stack":
            assert state.holding == ob and underob in state.clear
            new_state.on[ob] = underob
            new_state.clear.add(ob)
            new_state.clear.remove(underob)
            new_state.holding = None

        elif action_type == "unstack":
            assert ob in state.clear and state.on[ob] == underob and state.arm_empty
            new_state.holding = ob
            new_state.clear.remove(ob)
            new_state.clear.add(underob)
            new_state.on.pop(ob)

        return new_state

    def encode_state(self, state: BlockState) -> Dict[str, int]:
        """Encode state into binary features"""
        # Initialize all features to 0
        encoding = {name: 0 for name in self.feature_names}

        # Set features for blocks on table
        for block in state.on_table:
            encoding[f"{block}_on_table"] = 1

        # Set features for blocks on other blocks
        for block, under in state.on.items():
            encoding[f"{block}_on_{under}"] = 1

        # Set features for clear blocks
        for block in state.clear:
            encoding[f"{block}_clear"] = 1

        # Set feature for held block
        if state.holding:
            encoding[f"{state.holding}_held"] = 1

        return encoding

    def encode_state_as_vector(self, state: BlockState) -> List[int]:
        """Encode state as a binary vector for ML input"""
        encoding = self.encode_state(state)
        return [encoding[name] for name in self.feature_names]

    def decode_vector(self, vector: torch.Tensor) -> Dict[str, int]:
        """Decode binary vector to state encoding"""
        return dict(zip(self.feature_names, vector.tolist()))

    def decode_state(self, encoding: Dict[str, int]) -> BlockState:
        """Decode binary features back to BlockState"""
        clear = set()
        on_table = set()
        on = {}
        holding = None

        # Process each feature
        for feature, value in encoding.items():
            if value == 1:
                if "_on_table" in feature:
                    block = feature.split("_")[0]
                    on_table.add(block)
                elif "_on_" in feature:
                    block, _, under = feature.split("_")
                    on[block] = under
                elif "_clear" in feature:
                    block = feature.split("_")[0]
                    clear.add(block)
                elif "_held" in feature:
                    block = feature.split("_")[0]
                    holding = block

        return BlockState(clear=clear, on_table=on_table, holding=holding, on=on)

    def decode_vector_to_blocks(self, vector: torch.Tensor) -> BlockState:
        return self.decode_state(self.decode_vector(vector))

    def generate_random_plan(self) -> Dict | None:
        """Generate a random valid plan"""
        initial_state = self.generate_random_state()
        goal_state = self.generate_random_state()

        # Use breadth-first search to find a plan
        visited = set()
        queue = [(initial_state, [])]

        while queue:
            current_state, plan = queue.pop(0)

            state_hash = (
                frozenset(current_state.clear),
                frozenset(current_state.on_table),
                current_state.holding,
                frozenset(current_state.on.items()),
            )

            if (
                current_state.on == goal_state.on
                and current_state.on_table == goal_state.on_table
                and current_state.arm_empty
            ):
                # Convert plan to encoded states
                encoded_plan = []
                state = initial_state
                encoded_plan.append(self.encode_state_as_vector(state))

                for action in plan:
                    state = self.apply_action(state, action)
                    encoded_plan.append(self.encode_state_as_vector(state))

                return {
                    "initial_state": self.encode_state_as_vector(initial_state),
                    "goal_state": self.encode_state_as_vector(goal_state),
                    "plan": encoded_plan,
                    "actions": plan,
                    "feature_names": self.feature_names,
                }

            if state_hash in visited:
                continue

            visited.add(state_hash)

            for action in self.get_valid_actions(current_state):
                new_state = self.apply_action(current_state, action)
                queue.append((new_state, plan + [action]))

        return None

    def generate_dataset(self, num_plans: int) -> Dict:
        """Generate multiple random plans"""
        plans = []
        while len(plans) < num_plans:
            plan = self.generate_random_plan()
            if plan:
                plans.append(plan)
                print(f"Plan number {len(plans)} added.", end="\r")

        return {
            "plans": plans,
            "feature_names": self.feature_names,
            "num_features": len(self.feature_names),
        }
