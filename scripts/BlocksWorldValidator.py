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
        # self.held_indices = {}
        # self.arm_empty_index will be a single integer

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

        # Arm empty index
        self.arm_empty_index = idx
        idx += 1

        # Held indices
        self.held_indices = {}  # Initialize
        for block in blocks:
            self.held_indices[block] = idx
            idx += 1

        self.state_size = idx

    def _check_physical_constraints(self, state: List[int]) -> Tuple[bool, List[str]]:
        """Check if state satisfies basic physical constraints"""
        violations = []

        # Check state vector size
        if len(state) != self.state_size:
            violations.append(f"Invalid state size: expected {self.state_size}, got {len(state)}")
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
                violations.append(f"Block {block} is floating (not on any surface or held)")
            elif positions > 1:
                violations.append(f"Block {block} is in multiple positions simultaneously")

            # Check for cycles in stacking
            if self._has_cycle(state, block, set()):
                violations.append(f"Found cycle in block stacking involving block {block}")

            # Check clear status consistency
            if state[self.clear_indices[block]] == 1:
                for other in blocks:
                    if other != block and (other, block) in self.on_block_indices:
                        if state[self.on_block_indices[(other, block)]] == 1:
                            violations.append(f"Block {block} marked as clear but has {other} on top")

            # Check arm-empty and holding consistency
            num_held_blocks = 0
            for block_char_code in range(ord("A"), ord("A") + self.num_blocks):
                block = chr(block_char_code)
                if state[self.held_indices[block]] == 1:
                    num_held_blocks += 1

            is_arm_empty_predicate_true = state[self.arm_empty_index] == 1

            if num_held_blocks > 1:
                violations.append("Arm is holding multiple blocks.")

            if is_arm_empty_predicate_true:
                if num_held_blocks > 0:
                    violations.append("Arm-empty predicate is true, but arm is holding block(s).")
            else:  # Arm-empty predicate is false (i.e., arm should be holding something)
                if num_held_blocks == 0:
                    violations.append("Arm-empty predicate is false, but arm is not holding any block.")
                # If num_held_blocks > 1, it's already caught above.
                # If num_held_blocks == 1, this is consistent.

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

    def _check_legal_transition(self, state1: List[int], state2: List[int]) -> Tuple[bool, List[str]]:
        """Check if transition between states is legal according to blocks world rules.
        Assumes state1 and state2 are individually physically valid."""
        violations = []
        differences = sum(1 for a, b in zip(state1, state2) if a != b)

        if differences == 0:
            violations.append("No change between states")
        # A single action (pickup, putdown) changes 4 features.
        # A single action (stack, unstack) changes 5 features.
        elif differences < 4 or differences > 5:
            violations.append(
                f"Illegal number of changes ({differences} bits changed). Expected 4 or 5 for a single action, or 0 if at goal."
            )

        return len(violations) == 0, violations

    def validate_sequence(self, states: List[List[int]], goal_state: List[int]) -> ValidationResult:
        """Validate a complete sequence of states leading to a goal"""
        all_violations = []
        metrics = {}

        if not states:
            all_violations.append("Predicted sequence is empty.")
            # Assuming empty sequence doesn't achieve a typical BlocksWorld goal.
            # If goal_state could represent an "empty" or initial state, this might need adjustment.
            # For now, if sequence is empty, goal is not achieved unless goal_state is also somehow "empty".
            # is_empty_goal = not np.any(goal_state)  # Simplistic check if goal is all zeros (e.g. -1 after scaling)
            # Or, more robustly, define what an "empty" goal means.
            # For this problem, goal_state is unlikely to be "empty" in a way that matches an empty plan.

            metrics["sequence_length"] = 0
            metrics["goal_achievement"] = 0.0  # Default for empty sequence
            metrics["avg_changes_per_step"] = 0.0
            return ValidationResult(is_valid=len(all_violations) == 0, violations=all_violations, metrics=metrics)

        # 1. Validate all states individually for physical constraints
        for i, state_vector in enumerate(states):
            is_physically_valid, physical_violations = self._check_physical_constraints(state_vector)
            if not is_physically_valid:
                all_violations.extend([f"State {i} physically invalid: {v}" for v in physical_violations])

        # If any state is physically invalid, the sequence is fundamentally flawed.
        if all_violations:
            metrics["sequence_length"] = len(states)
            metrics["goal_achievement"] = 0.0
            metrics["avg_changes_per_step"] = 0.0  # Not meaningful if states are invalid
            return ValidationResult(is_valid=False, violations=all_violations, metrics=metrics)

        # 2. All states are physically valid. Now check transitions.
        for i in range(1, len(states)):
            # Note: _check_legal_transition now assumes states are physically valid.
            valid_transition, transition_violations_list = self._check_legal_transition(states[i - 1], states[i])

            if not valid_transition:
                is_acceptable_no_change = False
                # Check if the only violation is "No change between states" AND the state is the goal state
                if "No change between states" in transition_violations_list:
                    if np.array_equal(states[i - 1], goal_state):  # If the state that didn't change is the goal
                        # And "No change" is the *only* violation from _check_legal_transition
                        if (
                            len(transition_violations_list) == 1
                            and transition_violations_list[0] == "No change between states"
                        ):
                            is_acceptable_no_change = True

                if not is_acceptable_no_change:
                    all_violations.extend([f"Transition {i - 1}->{i}: {v}" for v in transition_violations_list])

        # 3. Check if the final state of the sequence matches the goal state
        final_state_is_goal = np.array_equal(states[-1], goal_state)
        if not final_state_is_goal:
            # Avoid overly verbose goal state printing if it's long
            # final_state_str = str(states[-1][:10]) + "..." if len(states[-1]) > 10 else str(states[-1])
            # goal_state_str = str(goal_state[:10]) + "..." if len(goal_state) > 10 else str(goal_state)
            all_violations.append("Final state does not match goal state.")

        # Populate metrics
        metrics["sequence_length"] = len(states)
        metrics["goal_achievement"] = float(final_state_is_goal)

        if len(states) > 1:
            sum_changes_for_avg = 0
            for i_tc in range(len(states) - 1):
                diff_count = sum(1 for a, b in zip(states[i_tc], states[i_tc + 1]) if a != b)
                sum_changes_for_avg += diff_count
            metrics["avg_changes_per_step"] = sum_changes_for_avg / (len(states) - 1)
        else:  # single state sequence
            metrics["avg_changes_per_step"] = 0.0

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
            if len(pred) == len(target) and all(np.array_equal(p, t) for p, t in zip(pred, target)):
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
    validator = BlocksWorldValidator(num_blocks=2)  # state_size should be 9
    print(f"Validator for 2 blocks expects state size: {validator.state_size}")

    # fmt: off

    # Valid state: A on table, B on A, A not clear, B clear, arm empty
    # Indices for 2 blocks (A, B) with arm-empty:
    # on_table_indices: A=0, B=1
    # on_block_indices: (A,B)=2, (B,A)=3
    # clear_indices: A=4, B=5
    # arm_empty_index: 6
    # held_indices: A=7, B=8
    valid_state = [
        1, 0,  # A on table, B not
        0, 1,  # A not on B, B on A
        0, 1,  # A not clear, B clear
        1,      # Arm empty
        0, 0,  # A not held, B not held
    ]
    # Corrected: A on table (1), B not on table (0). B on A (1). A not clear (0), B clear (1). Arm empty (1). Nothing held (0,0).
    # valid_state = [1,0, 0,1, 0,1, 1, 0,0] # This matches the derivation

    # Invalid state: A floating, arm empty (but A is not on anything or held)
    invalid_state_floating_A = [
        0, 0,  # A not on table, B not on table
        0, 0,  # A not on B, B not on A
        1, 1,  # A clear, B clear
        1,      # Arm empty
        0, 0,  # A not held, B not held
    ]

    # Invalid state: Arm empty, but A is held
    invalid_state_arm_empty_A_held = [
        0, 1,  # A not on table, B on table
        0, 0,  # No on_block relations
        0, 1,  # A not clear (because held), B clear
        1,      # Arm empty (Error!)
        1, 0,  # A held (Error!), B not held
    ]

    # fmt: on

    valid, violations = validator._check_physical_constraints(valid_state)
    print(f"Valid state check - Is valid: {valid}, Violations: {violations}")

    valid, violations = validator._check_physical_constraints(invalid_state_floating_A)
    print(f"Invalid state (floating A) check - Is valid: {valid}, Violations: {violations}")

    valid, violations = validator._check_physical_constraints(invalid_state_arm_empty_A_held)
    print(f"Invalid state (arm empty, A held) check - Is valid: {valid}, Violations: {violations}")

    # Test sequence validation
    # State 1: A on table, B on table, A clear, B clear, arm empty
    s1 = [1, 1, 0, 0, 1, 1, 1, 0, 0]
    # Action: pickup A
    # State 2: B on table, B clear, arm NOT empty, A held
    s2 = [0, 1, 0, 0, 0, 1, 0, 1, 0]  # A not clear, B clear, arm not empty, A held, B not on table

    sequence = [s1, s2]
    goal_state = s2

    result = validator.validate_sequence(sequence, goal_state)
    print("\nSequence validation result:")
    print(f"Is valid: {result.is_valid}")
    print(f"Violations: {result.violations}")
    print(f"Metrics: {result.metrics}")
