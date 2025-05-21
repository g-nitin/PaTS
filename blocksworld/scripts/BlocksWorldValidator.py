from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np


@dataclass
class ValidationResult:
    is_valid: bool
    violations: List[str]
    metrics: Dict[str, float]


class BlocksWorldValidator:
    def __init__(self, num_blocks: int):
        """Initialize validator for a specific number of blocks"""
        self.num_blocks = num_blocks
        self._setup_feature_indices()

    def _setup_feature_indices(self):
        """Calculate indices for different features in the state vector"""
        blocks = [chr(ord("A") + i) for i in range(self.num_blocks)]

        # Create maps for easy index lookup
        self.on_table_indices = {}
        self.on_block_indices = {}
        self.clear_indices = {}
        self.held_indices = {}

        idx = 0
        # On table indices
        for block in blocks:
            self.on_table_indices[block] = idx
            idx += 1

        # Block on block indices
        for block1 in blocks:
            for block2 in blocks:
                if block1 != block2:
                    self.on_block_indices[(block1, block2)] = idx
                    idx += 1

        # Clear indices
        for block in blocks:
            self.clear_indices[block] = idx
            idx += 1

        # Held indices
        for block in blocks:
            self.held_indices[block] = idx
            idx += 1

        self.state_size = idx

    def _check_physical_constraints(self, state: List[int]) -> Tuple[bool, List[str]]:
        """Check if state satisfies basic physical constraints"""
        violations = []

        # Check state vector size
        if len(state) != self.state_size:
            violations.append(
                f"Invalid state size: expected {self.state_size}, got {len(state)}"
            )
            return False, violations

        blocks = [chr(ord("A") + i) for i in range(self.num_blocks)]

        # Check each block
        for block in blocks:
            # Count number of positions (should be on exactly one surface or held)
            positions = 0
            if state[self.on_table_indices[block]] == 1:
                positions += 1
            for other in blocks:
                if other != block and (block, other) in self.on_block_indices:
                    if state[self.on_block_indices[(block, other)]] == 1:
                        positions += 1
            if state[self.held_indices[block]] == 1:
                positions += 1

            if positions == 0:
                violations.append(
                    f"Block {block} is floating (not on any surface or held)"
                )
            elif positions > 1:
                violations.append(
                    f"Block {block} is in multiple positions simultaneously"
                )

            # Check for cycles in stacking
            if self._has_cycle(state, block, set()):
                violations.append(
                    f"Found cycle in block stacking involving block {block}"
                )

            # Check clear status consistency
            if state[self.clear_indices[block]] == 1:
                for other in blocks:
                    if other != block and (other, block) in self.on_block_indices:
                        if state[self.on_block_indices[(other, block)]] == 1:
                            violations.append(
                                f"Block {block} marked as clear but has {other} on top"
                            )

        return len(violations) == 0, violations

    def _has_cycle(self, state: List[int], start_block: str, visited: Set[str]) -> bool:
        """Check for cycles in block stacking"""
        if start_block in visited:
            return True

        visited.add(start_block)
        blocks = [chr(ord("A") + i) for i in range(self.num_blocks)]

        for block in blocks:
            if block != start_block and (start_block, block) in self.on_block_indices:
                if state[self.on_block_indices[(start_block, block)]] == 1:
                    if self._has_cycle(state, block, visited):
                        return True

        visited.remove(start_block)
        return False

    def _check_legal_transition(
        self, state1: List[int], state2: List[int]
    ) -> Tuple[bool, List[str]]:
        """Check if transition between states is legal according to blocks world rules"""
        violations = []

        # First check both states are valid
        valid1, v1 = self._check_physical_constraints(state1)
        valid2, v2 = self._check_physical_constraints(state2)
        if not valid1:
            violations.extend([f"Initial state invalid: {v}" for v in v1])
            return False, violations
        if not valid2:
            violations.extend([f"Final state invalid: {v}" for v in v2])
            return False, violations

        # Count differences between states
        differences = sum(1 for a, b in zip(state1, state2) if a != b)

        # Legal moves involve picking up or putting down one block
        # This should change 2-4 bits in the state vector
        if differences == 0:
            violations.append("No change between states")
        elif differences > 4:
            violations.append(
                f"Too many changes between states ({differences} bits changed)"
            )

        return len(violations) == 0, violations

    def validate_sequence(
        self, states: List[List[int]], goal_state: List[int]
    ) -> ValidationResult:
        """Validate a complete sequence of states leading to a goal"""
        all_violations = []
        metrics = {}

        # Check each state and transition
        for i in range(len(states)):
            # Check current state
            valid, violations = self._check_physical_constraints(states[i])
            if not valid:
                all_violations.extend([f"State {i}: {v}" for v in violations])

            # Check transition from previous state
            if i > 0:
                valid, violations = self._check_legal_transition(
                    states[i - 1], states[i]
                )
                if not valid:
                    all_violations.extend(
                        [f"Transition {i - 1}->{i}: {v}" for v in violations]
                    )

        # Check if goal is reached
        if not np.array_equal(states[-1], goal_state):
            all_violations.append("Final state does not match goal state")

        # Calculate metrics
        metrics["sequence_length"] = len(states)
        metrics["goal_achievement"] = float(np.array_equal(states[-1], goal_state))
        if len(states) > 1:
            metrics["avg_changes_per_step"] = np.mean(
                [
                    sum(1 for a, b in zip(states[i], states[i + 1]) if a != b)
                    for i in range(len(states) - 1)
                ]
            )

        return ValidationResult(
            is_valid=len(all_violations) == 0,
            violations=all_violations,
            metrics=metrics,
        )

    def compute_performance_metrics(
        self,
        predictions: List[List[List[int]]],
        targets: List[List[List[int]]],
        goal_states: List[List[int]],
    ) -> Dict[str, float]:
        """Compute comprehensive performance metrics for a batch of predictions"""
        metrics = {
            "valid_sequence_rate": 0.0,
            "goal_achievement_rate": 0.0,
            "avg_sequence_length": 0.0,
            "avg_changes_per_step": 0.0,
            "exact_match_rate": 0.0,
        }

        num_samples = len(predictions)
        valid_sequences = 0
        goal_achieved = 0
        total_length = 0
        total_changes = 0
        exact_matches = 0

        for pred, target, goal in zip(predictions, targets, goal_states):
            # Validate prediction sequence
            result = self.validate_sequence(pred, goal)

            if result.is_valid:
                valid_sequences += 1
            if result.metrics["goal_achievement"] == 1.0:
                goal_achieved += 1
            total_length += result.metrics["sequence_length"]
            if "avg_changes_per_step" in result.metrics:
                total_changes += result.metrics["avg_changes_per_step"]

            # Check for exact match with target sequence
            if len(pred) == len(target) and all(
                np.array_equal(p, t) for p, t in zip(pred, target)
            ):
                exact_matches += 1

        # Calculate final metrics
        metrics["valid_sequence_rate"] = valid_sequences / num_samples
        metrics["goal_achievement_rate"] = goal_achieved / num_samples
        metrics["avg_sequence_length"] = total_length / num_samples
        metrics["avg_changes_per_step"] = total_changes / num_samples
        metrics["exact_match_rate"] = exact_matches / num_samples

        return metrics


# Example usage
if __name__ == "__main__":
    # Example with 2 blocks
    validator = BlocksWorldValidator(num_blocks=2)

    # Valid state: A on table, B on A, A not clear, B clear
    valid_state = [
        1,
        0,  # A,B on table
        0,
        1,  # A on B, B on A
        0,
        1,  # A,B clear
        0,
        0,
    ]  # A,B held

    # Invalid state: A floating
    invalid_state = [
        0,
        0,  # A,B on table
        0,
        0,  # A on B, B on A
        1,
        1,  # A,B clear
        0,
        0,
    ]  # A,B held

    # Test physical constraints
    valid, violations = validator._check_physical_constraints(valid_state)
    print(f"Valid state check - Is valid: {valid}, Violations: {violations}")

    valid, violations = validator._check_physical_constraints(invalid_state)
    print(f"Invalid state check - Is valid: {valid}, Violations: {violations}")

    # Test sequence validation
    sequence = [
        valid_state,
        # Next state: Both blocks on table
        [1, 1, 0, 0, 1, 1, 0, 0],
    ]
    goal_state = [1, 1, 0, 0, 1, 1, 0, 0]

    result = validator.validate_sequence(sequence, goal_state)
    print("\nSequence validation result:")
    print(f"Is valid: {result.is_valid}")
    print(f"Violations: {result.violations}")
    print(f"Metrics: {result.metrics}")
    print(f"Metrics: {result.metrics}")
