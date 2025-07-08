import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class Violation:
    code: str  # e.g., "PHYS_BLOCK_FLOATING", "TRANS_ILLEGAL_CHANGES"
    message: str
    details: Optional[Dict[str, Any]] = None  # e.g., {"block": "A"}


@dataclass
class ValidationResult:
    is_valid: bool
    violations: List[Violation]  # Changed from List[str]
    metrics: Dict[str, float]
    # New metrics
    goal_jaccard_score: float = 0.0
    goal_f1_score: float = 0.0
    # Add a field to store the predicted plan itself for easier aggregation later
    predicted_plan_length: int = 0


class BlocksWorldValidator:
    def __init__(self, num_blocks: int, encoding_type: str, predicate_manifest_file: Optional[str | Path] = None):
        """
        Initialize validator for a specific number of blocks and encoding type.

        :param num_blocks: The number of blocks in the domain.
        :param encoding_type: The encoding type, either 'binary' or 'sas'.
        :param predicate_manifest_file: Path to the predicate manifest. Required for 'binary' encoding.
        """
        self.num_blocks = num_blocks
        self.encoding_type = encoding_type
        self.block_names: List[str] = [f"b{i + 1}" for i in range(self.num_blocks)]

        if self.encoding_type == "binary":
            if predicate_manifest_file is None:
                raise ValueError("predicate_manifest_file is required for 'binary' encoding.")
            self.predicate_manifest_file = Path(predicate_manifest_file)
            # These will be populated by _setup_feature_indices_binary
            self.predicate_list: List[str] = []
            self.on_table_indices: Dict[str, int] = {}
            self.on_block_indices: Dict[Tuple[str, str], int] = {}
            self.clear_indices: Dict[str, int] = {}
            self.held_indices: Dict[str, int] = {}
            self.arm_empty_index: Optional[int] = None
            self._on_table_idx_list: List[int] = []
            self._on_block_idx_list: List[int] = []
            self._clear_idx_list: List[int] = []
            self._held_idx_list: List[int] = []
            self._setup_feature_indices_binary()
            self.state_size = len(self.predicate_list)
        elif self.encoding_type == "sas":
            self.state_size = self.num_blocks
        else:
            raise ValueError(f"Unsupported encoding_type: {encoding_type}")

    def _setup_feature_indices_binary(self):
        """Calculate indices for binary predicate encoding."""
        if not self.predicate_manifest_file.is_file():
            raise FileNotFoundError(f"Predicate manifest file not found: {self.predicate_manifest_file}")

        with open(self.predicate_manifest_file, "r") as f:
            self.predicate_list = [line.strip() for line in f if line.strip()]

        self.state_size = len(self.predicate_list)
        if self.state_size == 0:
            raise ValueError(f"Predicate manifest file {self.predicate_manifest_file} is empty or invalid.")

        # Determine block names based on num_blocks, assuming "bX" convention from parse_and_encode.py
        self.block_names = [f"b{i + 1}" for i in range(self.num_blocks)]

        for idx, pred_string_from_manifest in enumerate(self.predicate_list):
            pred_string = pred_string_from_manifest.lower()  # Already normalized by parser

            # On table: (on-table bX)
            m_on_table = re.fullmatch(r"\(on-table (b\d+)\)", pred_string)
            if m_on_table:
                block_name = m_on_table.group(1)
                if block_name in self.block_names:
                    self.on_table_indices[block_name] = idx
                continue

            # On block: (on bX bY)
            m_on_block = re.fullmatch(r"\(on (b\d+) (b\d+)\)", pred_string)
            if m_on_block:
                b1 = m_on_block.group(1)
                b2 = m_on_block.group(2)
                if b1 in self.block_names and b2 in self.block_names and b1 != b2:
                    self.on_block_indices[(b1, b2)] = idx
                continue

            # Clear: (clear bX)
            m_clear = re.fullmatch(r"\(clear (b\d+)\)", pred_string)
            if m_clear:
                block_name = m_clear.group(1)
                if block_name in self.block_names:
                    self.clear_indices[block_name] = idx
                continue

            # Arm empty: (arm-empty)
            if pred_string == "(arm-empty)":
                self.arm_empty_index = idx
                continue

            # Holding: (holding bX)
            m_holding = re.fullmatch(r"\(holding (b\d+)\)", pred_string)
            if m_holding:
                block_name = m_holding.group(1)
                if block_name in self.block_names:
                    self.held_indices[block_name] = idx
                continue

        # Populate the index lists for tensor operations
        self._on_table_idx_list = list(self.on_table_indices.values())
        self._on_block_idx_list = list(self.on_block_indices.values())
        self._clear_idx_list = list(self.clear_indices.values())
        self._held_idx_list = list(self.held_indices.values())

        # Sanity check: ensure arm_empty_index was found
        if self.arm_empty_index is None:
            raise ValueError("(arm-empty) predicate not found in manifest. This is required.")

    def calculate_constraint_violation_loss(self, state_logits: torch.Tensor) -> torch.Tensor:
        """Dispatch to the correct loss function based on encoding."""
        if self.encoding_type == "binary":
            return self._calculate_constraint_violation_loss_binary(state_logits)
        elif self.encoding_type == "sas":
            # NOTE: A differentiable loss for SAS+ is non-trivial as the values are discrete and interdependent.
            # A proper implementation might use a cross-entropy loss over possible locations for each block.
            # For now, we return zero loss, effectively disabling this feature for SAS+.
            return torch.tensor(0.0, device=state_logits.device)
        return torch.tensor(0.0, device=state_logits.device)

    def _calculate_constraint_violation_loss_binary(self, state_logits: torch.Tensor) -> torch.Tensor:
        """
        Calculates a differentiable loss term based on physical constraint violations.
        Operates on a batch of state logits from the model.

        :param state_logits: A tensor of shape (N, F) where N is the number of states in the batch and F is the number of features.
        :return: A scalar tensor representing the mean violation loss per state.
        """
        if state_logits.shape[0] == 0:
            return torch.tensor(0.0, device=state_logits.device)

        # Use probabilities for a smoother loss landscape
        state_probs = torch.sigmoid(state_logits)
        total_loss = torch.tensor(0.0, device=state_probs.device)
        num_states = state_probs.shape[0]

        # 1. PHYS_BLOCK_MULTI_POS: Each block must be in exactly one position.
        # (on table, on another block, or held).
        for block_name in self.block_names:
            pos_indices = []
            if block_name in self.on_table_indices:
                pos_indices.append(self.on_table_indices[block_name])
            if block_name in self.held_indices:
                pos_indices.append(self.held_indices[block_name])
            for b1, b2 in self.on_block_indices.keys():
                if b1 == block_name:
                    pos_indices.append(self.on_block_indices[(b1, b2)])

            # Sum the probabilities of the block being in any position
            pos_probs_sum = state_probs[:, pos_indices].sum(dim=1)
            # The sum should be 1.0. Penalize deviation from 1.0.
            total_loss += F.mse_loss(pos_probs_sum, torch.ones_like(pos_probs_sum))

        # 2. PHYS_CLEAR_CONFLICT: A block marked 'clear' cannot have another block on it.
        for block_name in self.block_names:
            clear_prob = state_probs[:, self.clear_indices[block_name]]

            # Sum probabilities of any other block being on top of this one
            on_top_indices = [
                self.on_block_indices[(other_block, block_name)]
                for other_block in self.block_names
                if other_block != block_name
            ]
            if on_top_indices:
                on_top_prob_sum = state_probs[:, on_top_indices].sum(dim=1)
                # Violation if both clear_prob and on_top_prob are high.
                # Their product should be 0.
                violation_prob = clear_prob * on_top_prob_sum
                total_loss += violation_prob.mean()

        # 3. PHYS_ARM_EMPTY_HOLDING_CONFLICT & PHYS_ARM_NOT_EMPTY_NOT_HOLDING_CONFLICT
        # Arm is empty IFF it is not holding any block.
        if self.arm_empty_index is not None and self._held_idx_list:
            arm_empty_prob = state_probs[:, self.arm_empty_index]
            is_holding_prob = state_probs[:, self._held_idx_list].sum(dim=1)

            # The sum of arm_empty_prob and is_holding_prob should be 1.0
            # e.g., if arm_empty is 0.9, is_holding should be 0.1.
            # This elegantly captures both conflict types.
            total_loss += F.mse_loss(arm_empty_prob + is_holding_prob, torch.ones_like(arm_empty_prob))

        return total_loss / num_states if num_states > 0 else torch.tensor(0.0)

    def _check_physical_constraints(self, state: List[int] | np.ndarray) -> Tuple[bool, List[Violation]]:
        """Dispatch to the correct physical constraint checker based on encoding."""
        if self.encoding_type == "binary":
            return self._check_physical_constraints_binary(state)
        elif self.encoding_type == "sas":
            return self._check_physical_constraints_sas(state)
        return False, [Violation("INTERNAL_ERROR", "Unknown encoding type in validator.")]

    def _check_physical_constraints_binary(self, state: List[int] | np.ndarray) -> Tuple[bool, List[Violation]]:
        """Check if state satisfies basic physical constraints"""
        violations: List[Violation] = []
        state_arr = np.array(state) if not isinstance(state, np.ndarray) else state

        if len(state_arr) != self.state_size:
            violations.append(
                Violation("STATE_INVALID_SIZE", f"Invalid state size: expected {self.state_size}, got {len(state_arr)}")
            )
            return False, violations

        # Check each block
        for block in self.block_names:
            positions = 0
            # On table
            if block in self.on_table_indices and state_arr[self.on_table_indices[block]] == 1:
                positions += 1
            # On another block
            for other_block in self.block_names:
                if other_block != block and (block, other_block) in self.on_block_indices:
                    if state_arr[self.on_block_indices[(block, other_block)]] == 1:
                        positions += 1
            # Held
            if block in self.held_indices and state_arr[self.held_indices[block]] == 1:
                positions += 1

            if positions == 0:
                violations.append(
                    Violation(
                        "PHYS_BLOCK_FLOATING",
                        f"Block {block} is floating (not on any surface or held)",
                        {"block": block},
                    )
                )
            elif positions > 1:
                violations.append(
                    Violation(
                        "PHYS_BLOCK_MULTI_POS",
                        f"Block {block} is in multiple positions simultaneously",
                        {"block": block},
                    )
                )

            # Check clear status consistency
            if block in self.clear_indices and state_arr[self.clear_indices[block]] == 1:
                for other_on_top in self.block_names:  # Check if any other_on_top is on 'block'
                    if other_on_top != block and (other_on_top, block) in self.on_block_indices:
                        if state_arr[self.on_block_indices[(other_on_top, block)]] == 1:
                            violations.append(
                                Violation(
                                    "PHYS_CLEAR_CONFLICT",
                                    f"Block {block} marked as clear but has {other_on_top} on top",
                                    {"block": block, "block_on_top": other_on_top},
                                )
                            )

        # Check arm-empty and holding consistency
        num_held_blocks = 0
        held_block_names = []
        for block_h in self.block_names:
            if block_h in self.held_indices and state_arr[self.held_indices[block_h]] == 1:
                num_held_blocks += 1
                held_block_names.append(block_h)

        if self.arm_empty_index is None:  # Should have been caught by _setup_feature_indices
            violations.append(Violation("INTERNAL_ERROR", "arm_empty_index not configured"))
            return False, violations  # Cannot proceed with this check

        is_arm_empty_predicate_true = state_arr[self.arm_empty_index] == 1

        if num_held_blocks > 1:
            violations.append(
                Violation(
                    "PHYS_ARM_MULTI_HOLD",
                    f"Arm is holding multiple blocks: {', '.join(held_block_names)}",
                    {"held_blocks": held_block_names},
                )
            )

        if is_arm_empty_predicate_true:
            if num_held_blocks > 0:
                violations.append(
                    Violation(
                        "PHYS_ARM_EMPTY_HOLDING_CONFLICT",
                        f"Arm-empty predicate is true, but arm is holding block(s): {', '.join(held_block_names)}",
                        {"held_blocks": held_block_names},
                    )
                )
        else:  # Arm-empty predicate is false (i.e., arm should be holding something)
            if num_held_blocks == 0:
                violations.append(
                    Violation(
                        "PHYS_ARM_NOT_EMPTY_NOT_HOLDING_CONFLICT",
                        "Arm-empty predicate is false, but arm is not holding any block.",
                    )
                )

        return len(violations) == 0, violations

    def _check_physical_constraints_sas(self, state: List[int] | np.ndarray) -> Tuple[bool, List[Violation]]:
        """Checks physical constraints for a SAS+ encoded state vector."""
        violations: List[Violation] = []
        state_arr = np.array(state, dtype=int) if not isinstance(state, np.ndarray) else state.astype(int)

        if len(state_arr) != self.state_size:
            violations.append(
                Violation("STATE_INVALID_SIZE", f"Invalid state size: expected {self.state_size}, got {len(state_arr)}")
            )
            return False, violations

        # 1. Check for multiple held blocks
        held_indices = np.where(state_arr == -1)[0]
        if len(held_indices) > 1:
            held_blocks = [self.block_names[i] for i in held_indices]
            violations.append(
                Violation(
                    "PHYS_ARM_MULTI_HOLD", f"Arm is holding multiple blocks: {held_blocks}", {"held_blocks": held_blocks}
                )
            )

        # 2. Check for invalid positions and build a "support" dictionary
        support_map = {}  # key=block_idx, value=supported_by_idx (0 for table, -1 for arm)
        for i, pos in enumerate(state_arr):
            if pos > self.num_blocks or pos < -1:
                violations.append(
                    Violation("PHYS_INVALID_POS", f"Block {self.block_names[i]} has invalid position value {pos}")
                )
            if pos == i + 1:
                violations.append(Violation("PHYS_SELF_SUPPORT", f"Block {self.block_names[i]} cannot be on itself."))
            if pos > 0:  # on another block
                support_map[i] = pos - 1  # store 0-based index of supporter
            elif pos == 0:  # on table
                support_map[i] = "table"
            elif pos == -1:  # held
                support_map[i] = "arm"

        # 3. Check for support cycles (e.g., b1 on b2, b2 on b1)
        for block_idx in range(self.num_blocks):
            path = [block_idx]
            curr = block_idx
            while curr in support_map and support_map[curr] not in ["table", "arm"]:
                curr = support_map[curr]
                if curr in path:
                    cycle_names = [self.block_names[p] for p in path] + [self.block_names[curr]]
                    violations.append(
                        Violation("PHYS_SUPPORT_CYCLE", f"Support cycle detected: {' -> '.join(cycle_names)}")
                    )
                    break  # Found a cycle, break inner while
                path.append(curr)

        # 4. Check for multiple blocks on the same support
        on_top_of = {}  # key=supporter_idx, value=list of blocks on top
        for i, pos in enumerate(state_arr):
            if pos > 0:  # on another block
                supporter_idx = pos - 1
                if supporter_idx not in on_top_of:
                    on_top_of[supporter_idx] = []
                on_top_of[supporter_idx].append(i)

        for supporter_idx, blocks_on_top in on_top_of.items():
            if len(blocks_on_top) > 1:
                supporter_name = self.block_names[supporter_idx]
                block_names_on_top = [self.block_names[b] for b in blocks_on_top]
                violations.append(
                    Violation("PHYS_MULTI_ON_TOP", f"Multiple blocks ({block_names_on_top}) on top of {supporter_name}")
                )

        return len(violations) == 0, violations

    def _check_legal_transition(self, state1: np.ndarray, state2: np.ndarray) -> Tuple[bool, List[Violation]]:
        """Dispatch to the correct transition checker based on encoding."""
        if self.encoding_type == "binary":
            return self._check_legal_transition_binary(state1, state2)
        elif self.encoding_type == "sas":
            return self._check_legal_transition_sas(state1, state2)
        return False, [Violation("INTERNAL_ERROR", "Unknown encoding type in validator.")]

    def _check_legal_transition_binary(self, state1: np.ndarray, state2: np.ndarray) -> Tuple[bool, List[Violation]]:
        """Check if transition between states is legal according to blocks world rules.
        Assumes state1 and state2 are individually physically valid."""
        violations: List[Violation] = []
        # Ensure states are numpy arrays for efficient comparison
        s1 = np.array(state1) if not isinstance(state1, np.ndarray) else state1
        s2 = np.array(state2) if not isinstance(state2, np.ndarray) else state2

        differences = np.sum(s1 != s2)

        if differences == 0:
            # This is not necessarily an error if the state is the goal state.
            # The validate_sequence method will handle this context.
            violations.append(Violation("TRANS_NO_CHANGE", "No change between states", {"diff_count": 0}))
        # A single action (pickup, putdown) changes 4 features.
        # A single action (stack, unstack) changes 5 features.
        elif differences < 4 or differences > 5:
            violations.append(
                Violation(
                    "TRANS_ILLEGAL_CHANGES",
                    f"Illegal number of changes ({differences} bits changed). Expected 4 or 5 for a single action.",
                    {"diff_count": int(differences)},
                )
            )

        return len(violations) == 0, violations

    def _check_legal_transition_sas(self, state1: np.ndarray, state2: np.ndarray) -> Tuple[bool, List[Violation]]:
        """Checks if transition between SAS+ states is legal."""
        violations: List[Violation] = []
        s1 = np.array(state1, dtype=int)
        s2 = np.array(state2, dtype=int)

        diff_indices = np.where(s1 != s2)[0]
        differences = len(diff_indices)

        if differences == 0:
            violations.append(Violation("TRANS_NO_CHANGE", "No change between states", {"diff_count": 0}))
        # A valid action (pickup, putdown, stack, unstack) changes the position of exactly ONE block.
        elif differences != 1:
            changed_blocks = [self.block_names[i] for i in diff_indices]
            violations.append(
                Violation(
                    "TRANS_ILLEGAL_CHANGES",
                    f"Illegal number of changes ({differences}). Expected 1 block to change position. Changed: {changed_blocks}",
                    {"diff_count": int(differences), "changed_blocks": changed_blocks},
                )
            )
        # Further checks could be added here to validate the specific change (e.g., was the moved block clear in s1?)
        # but for now, checking the number of changes is a strong heuristic.

        return len(violations) == 0, violations

    def validate_sequence(
        self, states: List[List[int] | np.ndarray], goal_state: List[int] | np.ndarray
    ) -> ValidationResult:
        """Validate a complete sequence of states leading to a goal"""
        all_violations_obj: List[Violation] = []
        metrics: Dict[str, float] = {}

        # Convert all states to numpy arrays for consistency
        np_states = [np.array(s) for s in states]
        np_goal_state = np.array(goal_state)

        if not np_states:
            all_violations_obj.append(Violation("SEQ_EMPTY", "Predicted sequence is empty."))
            metrics["sequence_length"] = 0
            metrics["goal_achievement"] = 0.0
            metrics["avg_changes_per_step"] = 0.0
            return ValidationResult(
                is_valid=False,
                violations=all_violations_obj,
                metrics=metrics,
                goal_jaccard_score=0.0,
                goal_f1_score=0.0,
                predicted_plan_length=0,
            )

        # 1. Validate all states individually for physical constraints
        physically_valid_states_count = 0
        for i, state_vector in enumerate(np_states):
            is_physically_valid, physical_violations = self._check_physical_constraints(state_vector)
            if not is_physically_valid:
                for v in physical_violations:
                    all_violations_obj.append(Violation(v.code, f"State {i} invalid: {v.message}", v.details))
            else:
                physically_valid_states_count += 1

        metrics["percent_physically_valid_states"] = (
            (physically_valid_states_count / len(np_states)) * 100 if np_states else 0.0
        )

        # If any state is physically invalid, the sequence is fundamentally flawed for some metrics,
        # but we can still report others.
        # Let's continue to check transitions if desired, or bail early.
        # For now, let's assume we continue to gather all possible violations.

        # 2. Check transitions.
        valid_transitions_count = 0
        if len(np_states) > 1:
            for i in range(len(np_states) - 1):
                # Pass np_states[i] and np_states[i+1]
                valid_transition, transition_violations_list = self._check_legal_transition(np_states[i], np_states[i + 1])

                is_acceptable_no_change = False
                if not valid_transition:
                    # Check if the only violation is "No change between states" AND the state is the goal state
                    if any(v.code == "TRANS_NO_CHANGE" for v in transition_violations_list):
                        if np.array_equal(np_states[i], np_goal_state):
                            if (
                                len(transition_violations_list) == 1
                                and transition_violations_list[0].code == "TRANS_NO_CHANGE"
                            ):
                                is_acceptable_no_change = True
                                # This "no change at goal" is fine, counts as a valid step in a sense
                                valid_transitions_count += 1

                    if not is_acceptable_no_change:
                        for v_trans in transition_violations_list:
                            all_violations_obj.append(
                                Violation(v_trans.code, f"Transition {i}->{i + 1}: {v_trans.message}", v_trans.details)
                            )
                else:
                    valid_transitions_count += 1
            metrics["percent_valid_transitions"] = (valid_transitions_count / (len(np_states) - 1)) * 100
        else:  # single state sequence
            metrics["percent_valid_transitions"] = 100.0  # Or 0.0, or N/A. If S0 is goal, it's valid.

        # 3. Check if the final state of the sequence matches the goal state
        final_state_is_goal = np.array_equal(np_states[-1], np_goal_state)
        if not final_state_is_goal:
            all_violations_obj.append(Violation("SEQ_GOAL_MISMATCH", "Final state does not match goal state."))

        metrics["goal_achievement"] = float(final_state_is_goal)

        # Calculate Jaccard and F1 for goal match
        final_pred_state = np_states[-1]
        pred_true_indices = set(np.where(final_pred_state == 1)[0])
        goal_true_indices = set(np.where(np_goal_state == 1)[0])

        intersection_len = len(pred_true_indices.intersection(goal_true_indices))
        union_len = len(pred_true_indices.union(goal_true_indices))

        jaccard = (
            intersection_len / union_len
            if union_len > 0
            else (1.0 if not pred_true_indices and not goal_true_indices else 0.0)
        )

        precision = (
            intersection_len / len(pred_true_indices)
            if len(pred_true_indices) > 0
            else (1.0 if intersection_len == 0 and len(goal_true_indices) == 0 else 0.0)
        )
        recall = (
            intersection_len / len(goal_true_indices)
            if len(goal_true_indices) > 0
            else (1.0 if intersection_len == 0 and len(pred_true_indices) == 0 else 0.0)
        )

        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else (1.0 if precision == 1.0 and recall == 1.0 else 0.0)
        )

        # Populate metrics
        metrics["sequence_length"] = len(np_states)

        if len(np_states) > 1:
            sum_changes_for_avg = 0
            for i_tc in range(len(np_states) - 1):
                diff_count = np.sum(np_states[i_tc] != np_states[i_tc + 1])
                sum_changes_for_avg += diff_count
            metrics["avg_changes_per_step"] = sum_changes_for_avg / (len(np_states) - 1)
        else:
            metrics["avg_changes_per_step"] = 0.0

        # Overall validity: no violations at all.
        # The definition of "is_valid" might need refinement.
        # For now, let's say it's valid if no *critical* violations.
        # Or, simply, if all_violations_obj is empty.
        # The original logic was: if all_violations is empty. Let's stick to that for now.
        # "TRANS_NO_CHANGE" at goal is acceptable and shouldn't make it invalid.
        # We need to filter out acceptable "TRANS_NO_CHANGE" violations before checking if all_violations_obj is empty.

        final_violations_for_is_valid_check = []
        for v_obj in all_violations_obj:
            if v_obj.code == "TRANS_NO_CHANGE":
                # Check if this "no change" occurred when the state was already the goal
                # This requires knowing which state index this transition violation refers to.
                # The message "Transition {i}->{i+1}" helps.
                match_trans_idx = re.search(r"Transition (\d+)->(\d+)", v_obj.message)
                if match_trans_idx:
                    state_idx_before_no_change = int(match_trans_idx.group(1))
                    if np.array_equal(np_states[state_idx_before_no_change], np_goal_state):
                        continue  # This "no change" is acceptable, don't count for overall invalidity
            final_violations_for_is_valid_check.append(v_obj)

        return ValidationResult(
            is_valid=len(final_violations_for_is_valid_check) == 0,
            violations=all_violations_obj,  # Return all, even if some are acceptable for is_valid
            metrics=metrics,
            goal_jaccard_score=jaccard,
            goal_f1_score=f1,
            predicted_plan_length=len(np_states),
        )
