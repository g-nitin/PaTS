import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class Violation:
    code: str  # e.g., "PHYS_BLOCK_FLOATING", "TRANS_ILLEGAL_CHANGES"
    message: str
    details: Optional[Dict[str, Any]] = None  # e.g., {"block": "A"}


@dataclass
class ValidationResult:
    is_valid: bool
    violations: List[Violation]
    metrics: Dict[str, float]
    goal_jaccard_score: float = 0.0
    goal_f1_score: float = 0.0
    predicted_plan_length: int = 0


class GrippersValidator:
    def __init__(self, encoding_type: str, processed_data_dir: Path):
        """
        Initialize validator for the Grippers domain.

        :param encoding_type: The encoding type, either 'bin' or 'sas'.
        :param processed_data_dir: Path to the processed data directory for this config and encoding.
        """
        self.encoding_type = encoding_type
        self.processed_data_dir = processed_data_dir

        # Load encoding info
        encoding_info_path = self.processed_data_dir / "encoding_info.json"
        if not encoding_info_path.is_file():
            raise FileNotFoundError(f"Encoding info file not found: {encoding_info_path}")
        with open(encoding_info_path, "r") as f:
            self.encoding_info = json.load(f)

        config = self.encoding_info["domain_config"]
        self.num_robots = config["robots"]
        self.num_objects = config["objects"]
        self.num_rooms = config["rooms"]
        self.state_size = self.encoding_info["feature_dim"]

        if self.encoding_type == "bin":
            self.predicate_manifest_file = self.processed_data_dir / self.encoding_info["manifest_file"]
            with open(self.predicate_manifest_file, "r") as f:
                self.predicate_list = [line.strip() for line in f if line.strip()]
        elif self.encoding_type == "sas":
            self.robot_names = self.encoding_info["entity_order"]["robots"]
            self.object_names = self.encoding_info["entity_order"]["objects"]
            self.gripper_map = self.encoding_info["gripper_map"]
        else:
            raise ValueError(f"Unsupported encoding_type: {encoding_type}")

    def calculate_constraint_violation_loss(self, state_logits: torch.Tensor) -> torch.Tensor:
        """Constraint loss is not implemented for Grippers. Returns zero."""
        # NOTE: This function can be implemented in the future for constraint-guided training.
        return torch.tensor(0.0, device=state_logits.device)

    def _check_physical_constraints(self, state: List[int] | np.ndarray) -> Tuple[bool, List[Violation]]:
        """Dispatch to the correct physical constraint checker based on encoding."""
        if self.encoding_type == "bin":
            return self._check_physical_constraints_binary(state)
        elif self.encoding_type == "sas":
            return self._check_physical_constraints_sas(state)
        return False, [Violation("INTERNAL_ERROR", "Unknown encoding type in validator.")]

    def _check_physical_constraints_binary(self, state: List[int] | np.ndarray) -> Tuple[bool, List[Violation]]:
        """Check if a binary state satisfies Grippers physical constraints."""
        violations: List[Violation] = []
        state_arr = np.array(state) if not isinstance(state, np.ndarray) else state

        if len(state_arr) != self.state_size:
            violations.append(Violation("STATE_INVALID_SIZE", f"Expected size {self.state_size}, got {len(state_arr)}"))
            return False, violations

        true_preds = {self.predicate_list[i] for i, val in enumerate(state_arr) if val == 1}

        # Rule 1: Each robot must be in exactly one room.
        for r_idx in range(1, self.num_robots + 1):
            loc_count = sum(1 for p in true_preds if p.startswith(f"(at-robby r{r_idx}"))
            if loc_count != 1:
                violations.append(
                    Violation("PHYS_ROBOT_POS_ERROR", f"Robot r{r_idx} is in {loc_count} rooms (expected 1).")
                )

        # Rule 2: Each object must be in exactly one place (room or gripper).
        for o_idx in range(1, self.num_objects + 1):
            loc_count = sum(
                1 for p in true_preds if f" ball{o_idx}" in p and (p.startswith("(at ") or p.startswith("(carry "))
            )
            if loc_count != 1:
                violations.append(
                    Violation("PHYS_OBJECT_POS_ERROR", f"Object ball{o_idx} is in {loc_count} locations (expected 1).")
                )

        # Rule 3: A gripper is either free or carrying, not both.
        for r_idx in range(1, self.num_robots + 1):
            for gripper in ["left", "right"]:
                is_free = f"(free r{r_idx} {gripper})" in true_preds
                is_carrying = any(p.startswith(f"(carry r{r_idx}") and p.endswith(f" {gripper})") for p in true_preds)
                if is_free and is_carrying:
                    violations.append(
                        Violation("PHYS_GRIPPER_CONFLICT", f"Gripper r{r_idx}-{gripper} is both free and carrying.")
                    )
                if not is_free and not is_carrying:
                    violations.append(
                        Violation("PHYS_GRIPPER_STATELESS", f"Gripper r{r_idx}-{gripper} is neither free nor carrying.")
                    )

        return len(violations) == 0, violations

    def _check_physical_constraints_sas(self, state: List[int] | np.ndarray) -> Tuple[bool, List[Violation]]:
        """Checks physical constraints for a Grippers SAS+ encoded state vector."""
        violations: List[Violation] = []
        state_arr = np.array(state, dtype=int)

        if len(state_arr) != self.state_size:
            violations.append(Violation("STATE_INVALID_SIZE", f"Expected size {self.state_size}, got {len(state_arr)}"))
            return False, violations

        robot_positions = state_arr[: self.num_robots]
        object_positions = state_arr[self.num_robots :]

        # Rule 1: Check robot positions are valid room IDs.
        for i, pos in enumerate(robot_positions):
            if not (1 <= pos <= self.num_rooms):
                violations.append(
                    Violation("PHYS_INVALID_ROBOT_POS", f"Robot {self.robot_names[i]} has invalid room position {pos}.")
                )

        # Rule 2: Check object positions are valid room or gripper IDs.
        valid_gripper_ids = set(self.gripper_map.values())
        for i, pos in enumerate(object_positions):
            is_in_room = 1 <= pos <= self.num_rooms
            is_in_gripper = pos in valid_gripper_ids
            if not (is_in_room or is_in_gripper):
                violations.append(
                    Violation("PHYS_INVALID_OBJECT_POS", f"Object {self.object_names[i]} has invalid position value {pos}.")
                )

        # Rule 3: Check that each gripper holds at most one object.
        held_positions = [p for p in object_positions if p < 0]
        if len(held_positions) != len(set(held_positions)):
            violations.append(Violation("PHYS_GRIPPER_MULTI_CARRY", "A single gripper is holding multiple objects."))

        return len(violations) == 0, violations

    def _check_legal_transition(self, state1: np.ndarray, state2: np.ndarray) -> Tuple[bool, List[Violation]]:
        """Dispatch to the correct transition checker based on encoding."""
        if self.encoding_type == "bin":
            return self._check_legal_transition_binary(state1, state2)
        elif self.encoding_type == "sas":
            return self._check_legal_transition_sas(state1, state2)
        return False, [Violation("INTERNAL_ERROR", "Unknown encoding type in validator.")]

    def _check_legal_transition_binary(self, state1: np.ndarray, state2: np.ndarray) -> Tuple[bool, List[Violation]]:
        """Check if transition is legal for Grippers. A single action changes 2 or 3 predicates."""
        violations: List[Violation] = []
        s1 = np.array(state1)
        s2 = np.array(state2)
        differences = np.sum(s1 != s2)

        if differences == 0:
            violations.append(Violation("TRANS_NO_CHANGE", "No change between states", {"diff_count": 0}))
        # move=2 changes, pick=3 changes, drop=3 changes
        elif differences not in [2, 3]:
            violations.append(
                Violation(
                    "TRANS_ILLEGAL_CHANGES",
                    f"Illegal number of changes ({differences}). Expected 2 or 3 for a single action.",
                    {"diff_count": int(differences)},
                )
            )
        return len(violations) == 0, violations

    def _check_legal_transition_sas(self, state1: np.ndarray, state2: np.ndarray) -> Tuple[bool, List[Violation]]:
        """Checks if transition between SAS+ states is legal. A single action changes exactly one entity's position."""
        violations: List[Violation] = []
        s1 = np.array(state1, dtype=int)
        s2 = np.array(state2, dtype=int)
        diff_indices = np.where(s1 != s2)[0]
        differences = len(diff_indices)

        if differences == 0:
            violations.append(Violation("TRANS_NO_CHANGE", "No change between states", {"diff_count": 0}))
        elif differences != 1:
            violations.append(
                Violation(
                    "TRANS_ILLEGAL_CHANGES",
                    f"Illegal number of changes ({differences}). Expected 1 entity to change position.",
                    {"diff_count": int(differences)},
                )
            )
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
