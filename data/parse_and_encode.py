import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pddlpy  # For parsing PDDL files
from pddlpy.pddl import Atom  # Import the Atom class


# ** Predicate List Generation (Crucial for Consistent Encoding) **
def get_ordered_predicate_list(num_blocks, block_names_prefix="b"):
    """
    Generates a fixed, ordered list of all possible predicates for a given number of blocks.
    This order MUST be consistent for all problems with the same num_blocks.
    Block names are generated as b1, b2, ...
    Predicates are normalized (lowercase, single space).
    """
    # Generate block names like b1, b2, b3...
    block_names = [f"{block_names_prefix}{i + 1}" for i in range(num_blocks)]
    predicates = []

    # on-table ?b - block
    for b_name in block_names:
        predicates.append(f"(on-table {b_name})")

    # on ?b1 ?b2 - block, block
    for b1_name in block_names:
        for b2_name in block_names:
            if b1_name == b2_name:
                continue
            predicates.append(f"(on {b1_name} {b2_name})")

    # clear ?b - block
    for b_name in block_names:
        predicates.append(f"(clear {b_name})")

    # arm-empty
    predicates.append("(arm-empty)")

    # holding ?b - block
    for b_name in block_names:
        predicates.append(f"(holding {b_name})")

    # The generated predicates are already in a normalized form by construction.
    return predicates


def state_preds_to_sas_vector(state_predicates_set, num_blocks, block_names):
    """
    Converts a set of true predicate strings (normalized) to a SAS+ position vector.
    Vector of size num_blocks.
    - Index i corresponds to block_names[i] (e.g., b1).
    - Value 0 means on the table.
    - Value j > 0 means on block_names[j-1] (e.g., value 2 means on b2).
    - Value -1 means held by the arm.
    """
    # Initialize vector with a value indicating "floating", which validator should catch.
    position_vector = np.full(num_blocks, -2, dtype=np.int8)

    # Map block names to their 1-based index for "on" relationships (e.g., b1 -> 1, b2 -> 2)
    block_to_sas_idx = {name: i + 1 for i, name in enumerate(block_names)}

    for pred in state_predicates_set:
        # (on-table bX)
        m_on_table = re.fullmatch(r"\(on-table (b\d+)\)", pred)
        if m_on_table:
            block_name = m_on_table.group(1)
            block_idx = block_names.index(block_name)
            position_vector[block_idx] = 0
            continue

        # (on bX bY)
        m_on_block = re.fullmatch(r"\(on (b\d+) (b\d+)\)", pred)
        if m_on_block:
            block_on_top = m_on_block.group(1)
            block_below = m_on_block.group(2)
            top_idx = block_names.index(block_on_top)
            below_sas_idx = block_to_sas_idx[block_below]
            position_vector[top_idx] = below_sas_idx
            continue

        # (holding bX)
        m_holding = re.fullmatch(r"\(holding (b\d+)\)", pred)
        if m_holding:
            block_name = m_holding.group(1)
            block_idx = block_names.index(block_name)
            position_vector[block_idx] = -1
            continue

    return position_vector


# ** Predicate String Normalization **
def normalize_predicate_string(p_str_unnormalized):
    """
    Normalizes a predicate string to a canonical form:
    "(predicate arg1 arg2)" (all lowercase, single internal spaces, trimmed, outer parens).
    Input is expected to be in a form like "(content)" e.g. "(on b1 b2)" or "(handempty)".
    """
    content = p_str_unnormalized.strip()

    # Ensure it's wrapped in parentheses if not already (e.g. "handempty" from VAL)
    if not (content.startswith("(") and content.endswith(")")):
        # VAL sometimes outputs "handempty" without parens.
        # The PDDL standard is (predicate ...).
        # Our master list uses (arm-empty). Let's be flexible with VAL's "handempty".
        if content.lower() == "handempty":
            content = "(arm-empty)"  # Normalize to our standard
        else:
            content = f"({content})"

    # Get content within the outermost parens and strip whitespace
    inner_content = content[1:-1].strip()

    if not inner_content:  # Handles "()" or "( )"
        return "()"

    parts = inner_content.lower().split()  # Split into predicate name and args, convert to lowercase

    # Normalize "handempty" to "arm-empty" if it appears as a predicate name
    if parts[0] == "handempty":
        parts[0] = "arm-empty"

    # Reconstruct with single spaces
    if len(parts) > 1:
        return f"({parts[0]} {' '.join(parts[1:])})"
    else:  # Nullary predicate
        return f"({parts[0]})"


# ** PDDLPY Helper Functions **
def pddlpy_pred_to_normalized_string(pred_input):
    """
    Converts a PDDLPY predicate representation (either an Atom object from pddl.py,
    or a tuple of strings e.g. from parsing a Literal object)
    to a normalized string (e.g., "(on-table b1)" or "(handempty)").
    """
    predicate_parts = []
    if isinstance(pred_input, Atom):  # Check if it's an Atom object
        # pred_input.predicate is a list like ['on-table', 'b1'] or ['handempty']
        predicate_parts = pred_input.predicate
    elif isinstance(pred_input, tuple):  # Tuple of strings like ('on-table', 'b1')
        predicate_parts = list(pred_input)
    else:
        raise TypeError(f"Unsupported predicate input type: {type(pred_input)}. Expected pddlpy.pddl.Atom or tuple.")

    if not predicate_parts:
        # This case should ideally not be reached if PDDL is valid and parsing is correct.
        # print(f"    WARNING: pddlpy_pred_to_normalized_string received empty predicate_parts for input: {pred_input}")
        return "()"

    pred_name = str(predicate_parts[0]).lower()
    # Normalize "handempty" to "arm-empty" at the source
    if pred_name == "handempty":
        pred_name = "arm-empty"

    pred_args = [str(arg).lower() for arg in predicate_parts[1:]]

    if pred_args:
        return f"({pred_name} {' '.join(pred_args)})"
    else:
        return f"({pred_name})"


def get_goal_preds_from_pddlpy(problem_goal_literal):
    """
    Extracts a set of normalized goal predicate strings from PDDLPY's goal structure.
    Supports conjunctive goals (nested ANDs). Other logical constructs are not supported.
    Input: domprob.problem.goal (a pddlpy.literal.Literal object)
    Output: set of normalized predicate strings
    """
    goal_predicate_tuples = set()

    def extract_conjunctive_goals_recursive(literal_node):
        if literal_node is None:  # Handle cases where a goal might be empty or malformed
            print("    WARNING: extract_conjunctive_goals_recursive received a None literal_node.")
            return

        # Assuming literal_node.op and literal_node.args exist as per pddlpy.literal.Literal structure
        op_name = str(literal_node.op).lower()

        if op_name == "and":
            for arg_literal in literal_node.args:
                extract_conjunctive_goals_recursive(arg_literal)
        elif op_name in ["or", "not", "imply", "forall", "exists"]:
            # If you encounter this, you might need to extend goal parsing or simplify PDDL goals.
            print(
                f"    WARNING: PaTS currently only supports simple conjunctive PDDL goals for 'get_goal_preds_from_pddlpy'. Found operator: {op_name}. Full goal: {literal_node}. "
                "The 'domprob.goals()' method might still provide positive literals if available."
            )
            # Optionally, try to fall back to domprob.goals() if this structured parsing fails for complex goals.
            # However, for now, we'll stick to raising awareness.
            # raise NotImplementedError(f"Unsupported PDDL goal operator: {op_name}")
        else:  # It's a simple predicate
            args_lower = tuple(str(a).lower() for a in literal_node.args)
            # Normalize predicate name here too if necessary, e.g., handempty -> arm-empty
            final_op_name = "arm-empty" if op_name == "handempty" else op_name
            goal_predicate_tuples.add((final_op_name,) + args_lower)

    extract_conjunctive_goals_recursive(problem_goal_literal)

    # Convert tuples to normalized strings
    return {pddlpy_pred_to_normalized_string(p_tuple) for p_tuple in goal_predicate_tuples}


# ** VAL Output Parser **
def extract_states_from_val_output(val_output_filepath, initial_state_preds_normalized_set):
    """
    Extracts state trajectory (list of lists of sorted, normalized predicate strings) from VAL output.
    initial_state_preds_normalized_set is a set of normalized predicate strings for S0.
    """
    state_trajectory_pred_strings_list = []

    current_state_normalized = set(initial_state_preds_normalized_set)
    # Add S0 (initial state) to the trajectory, as a sorted list of strings
    state_trajectory_pred_strings_list.append(list(sorted(list(current_state_normalized))))

    # Regex for "Deleting PRED_CONTENT" or "Adding PRED_CONTENT"
    # PRED_CONTENT can be "pred arg1 arg2" (possibly with parens) or just "pred"
    delta_predicate_regex = re.compile(r"^(?:Deleting|Adding)\s+(.*?)\s*$", re.IGNORECASE)

    try:
        with open(val_output_filepath, "r") as f:
            val_lines = f.readlines()
    except FileNotFoundError:
        print(f"ERROR: VAL output file not found at {val_output_filepath}")
        raise

    plan_details_started = False
    for line_content in val_lines:
        line = line_content.strip()

        if not plan_details_started:
            if "Plan Validation details" in line or "Checking instance" in line or "Initial state:" in line:
                plan_details_started = True
            if not plan_details_started:
                continue

        is_boundary = "Checking next happening" in line or "Plan executed successfully" in line or "Plan valid" in line

        if is_boundary:
            # Check if current_state_normalized is different from the last recorded state's set representation
            if (
                not state_trajectory_pred_strings_list
                or set(state_trajectory_pred_strings_list[-1]) != current_state_normalized
            ):
                state_trajectory_pred_strings_list.append(list(sorted(list(current_state_normalized))))

            if "Plan executed successfully" in line or "Plan valid" in line:
                break  # End of plan processing

        match = delta_predicate_regex.match(line)
        if match:
            predicate_content_from_val = match.group(1)
            normalized_pred = normalize_predicate_string(predicate_content_from_val)

            if line.lower().startswith("deleting"):
                if normalized_pred in current_state_normalized:
                    current_state_normalized.remove(normalized_pred)
                # else:
                #     print(f"    WARNING: VAL tried to delete non-existent predicate '{normalized_pred}' from current state in {val_output_filepath}")
            elif line.lower().startswith("adding"):
                current_state_normalized.add(normalized_pred)

    # Final check to ensure the very last state is captured if it changed after the last boundary
    if not state_trajectory_pred_strings_list:  # Should not happen if initial state was added
        if initial_state_preds_normalized_set:  # Add initial state if trajectory is somehow empty
            state_trajectory_pred_strings_list.append(list(sorted(list(initial_state_preds_normalized_set))))
    elif set(state_trajectory_pred_strings_list[-1]) != current_state_normalized:
        state_trajectory_pred_strings_list.append(list(sorted(list(current_state_normalized))))

    return state_trajectory_pred_strings_list


# ** State to Binary Vector **
def state_preds_to_binary_vector(state_predicates_set, ordered_master_predicate_list):
    """
    Converts a set of true predicate strings (normalized) to a binary vector
    based on the ordered_master_predicate_list (which must also contain normalized predicates).
    """
    binary_vector = np.zeros(len(ordered_master_predicate_list), dtype=np.int8)
    for i, p_template_normalized in enumerate(ordered_master_predicate_list):
        if p_template_normalized in state_predicates_set:
            binary_vector[i] = 1
    return binary_vector


def main():
    parser = argparse.ArgumentParser(description="Parse VAL output and PDDL to generate state trajectories.")
    parser.add_argument("--val_output_file", required=True, help="Path to VAL's verbose output file.")
    parser.add_argument("--pddl_domain_file", required=True, help="Path to the PDDL domain file (for pddlpy).")
    parser.add_argument("--pddl_problem_file", required=True, help="Path to the PDDL problem file.")
    parser.add_argument("--num_blocks", required=True, type=int, help="Number of blocks (for master predicate list).")
    parser.add_argument("--text_trajectory_output", required=True, help="Path to save text-based predicate trajectory.")
    parser.add_argument(
        "--binary_trajectory_output", required=True, help="Path to save binary encoded state trajectory (.npy)."
    )
    parser.add_argument("--binary_goal_output", required=True, help="Path to save binary encoded goal state (.npy).")
    parser.add_argument("--manifest_output_dir", required=True, help="Directory to save the predicate manifest file.")
    parser.add_argument(
        "--encoding_type",
        type=str,
        choices=["binary", "sas"],
        default="binary",
        help="The type of state encoding to use: 'binary' (predicate vector) or 'sas' (position vector).",
    )

    args = parser.parse_args()

    print(f"    INFO: Processing {args.pddl_problem_file} with {args.val_output_file}")
    print(f"    INFO: Using encoding type: {args.encoding_type}")

    try:
        # Generate block names once
        block_names = [f"b{i + 1}" for i in range(args.num_blocks)]

        # 1. Generate encoding-specific information
        if args.encoding_type == "binary":
            ordered_master_pred_list = get_ordered_predicate_list(args.num_blocks, block_names_prefix="b")
            if not ordered_master_pred_list:
                print(f"ERROR: Could not generate master predicate list for {args.num_blocks} blocks.")
                exit(1)

            # Save the predicate manifest
            manifest_dir = Path(args.manifest_output_dir)
            manifest_dir.mkdir(parents=True, exist_ok=True)
            manifest_file_path = manifest_dir / f"predicate_manifest_{args.num_blocks}.txt"
            with open(manifest_file_path, "w") as f_manifest:
                for pred_item in ordered_master_pred_list:
                    f_manifest.write(f"{pred_item}\n")
            print(f"    INFO: Predicate manifest saved to: {manifest_file_path}")
            encoding_info = {
                "type": "binary",
                "manifest_file": str(manifest_file_path.name),
                "feature_dim": len(ordered_master_pred_list),
            }
        elif args.encoding_type == "sas":
            # For SAS, the "manifest" is just the number of blocks.
            ordered_master_pred_list = None  # Not used for SAS
            encoding_info = {"type": "sas", "feature_dim": args.num_blocks, "block_order": block_names}
        else:
            raise ValueError(f"Unknown encoding type: {args.encoding_type}")

        # Save the encoding info file
        encoding_info_path = Path(args.manifest_output_dir) / f"encoding_info_{args.num_blocks}.json"
        with open(encoding_info_path, "w") as f_info:
            json.dump(encoding_info, f_info, indent=4)
        print(f"    INFO: Encoding info saved to: {encoding_info_path}")

        # 2. Parse PDDL domain and problem files using pddlpy
        try:
            domprob = pddlpy.DomainProblem(args.pddl_domain_file, args.pddl_problem_file)
        except Exception as e:
            print(f"ERROR: Failed to parse PDDL files with pddlpy: {e}")
            print(f"  Domain: {args.pddl_domain_file}")
            print(f"  Problem: {args.pddl_problem_file}")
            raise

        # Parse Initial State: domprob.initialstate() returns a set of pddlpy.pddl.Atom objects
        initial_state_atoms = domprob.initialstate()
        initial_state_preds_normalized_set = {pddlpy_pred_to_normalized_string(atom) for atom in initial_state_atoms}
        if not initial_state_preds_normalized_set and args.num_blocks > 0:
            print(f"    WARNING: No initial state predicates parsed from {args.pddl_problem_file}.")

        # Parse Goal State:
        # domprob.problem.goal is expected to be a pddlpy.literal.Literal object representing the goal structure
        # get_goal_preds_from_pddlpy processes this Literal tree.
        goal_state_preds_normalized_set = set()
        if hasattr(domprob, "problem") and hasattr(domprob.problem, "goal") and domprob.problem.goals is not None:
            goal_state_preds_normalized_set = get_goal_preds_from_pddlpy(domprob.problem.goals)
        else:
            # Fallback or alternative: domprob.goals() returns a set of Atom objects for simple goals
            # This might be useful if domprob.problem.goal is not populated or for simpler goal structures.
            print(
                f"    INFO: domprob.problem.goal not found or is None. Trying domprob.goals() for {args.pddl_problem_file}."
            )
            goal_atoms_from_goals_method = domprob.goals()  # set of Atom objects
            if goal_atoms_from_goals_method:
                goal_state_preds_normalized_set = {
                    pddlpy_pred_to_normalized_string(atom) for atom in goal_atoms_from_goals_method
                }
            else:
                print(f"    WARNING: No goal predicates parsed from {args.pddl_problem_file} using either method.")

        state_trajectory_pred_strings_list = extract_states_from_val_output(
            args.val_output_file, initial_state_preds_normalized_set
        )

        if not state_trajectory_pred_strings_list:  # Must have at least S0
            print(f"ERROR: Failed to extract any states from {args.val_output_file}. Trajectory is empty.")
            exit(1)

        # 4. Convert each state in the trajectory to a vector
        trajectory_vectors = []
        for state_preds_list_for_one_step in state_trajectory_pred_strings_list:
            state_preds_set_for_one_step = set(state_preds_list_for_one_step)
            if args.encoding_type == "binary":
                vector = state_preds_to_binary_vector(state_preds_set_for_one_step, ordered_master_pred_list)
            else:  # sas
                vector = state_preds_to_sas_vector(state_preds_set_for_one_step, args.num_blocks, block_names)
            trajectory_vectors.append(vector)
        trajectory_np = np.array(trajectory_vectors, dtype=np.int8)

        # 5. Convert goal state to vector
        if args.encoding_type == "binary":
            goal_vector_np = state_preds_to_binary_vector(goal_state_preds_normalized_set, ordered_master_pred_list)
        else:  # sas
            goal_vector_np = state_preds_to_sas_vector(goal_state_preds_normalized_set, args.num_blocks, block_names)

        # 6. Save outputs with new naming convention
        base_traj_path = Path(args.binary_trajectory_output)
        traj_output_path = base_traj_path.with_suffix(f".{args.encoding_type}.npy")

        base_goal_path = Path(args.binary_goal_output)
        goal_output_path = base_goal_path.with_suffix(f".{args.encoding_type}.npy")

        os.makedirs(os.path.dirname(traj_output_path), exist_ok=True)
        np.save(traj_output_path, trajectory_np)

        os.makedirs(os.path.dirname(goal_output_path), exist_ok=True)
        np.save(goal_output_path, goal_vector_np)

        traj_len = trajectory_np.shape[0] if trajectory_np.ndim > 0 and trajectory_np.size > 0 else 0
        vec_dim = trajectory_np.shape[1] if trajectory_np.ndim > 1 and traj_len > 0 else encoding_info["feature_dim"]

        print(f"    INFO: Successfully processed. Trajectory Length: {traj_len}. Vector dim: {vec_dim}")
        print(f"    INFO: Text trajectory saved to: {args.text_trajectory_output}")
        print(f"    INFO: Encoded trajectory saved to: {traj_output_path}")
        print(f"    INFO: Encoded goal saved to: {goal_output_path}")

    except Exception as e:
        print(f"FATAL ERROR in parse_and_encode.py: {e}")
        import traceback

        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
