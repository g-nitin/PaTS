import argparse
import os
import re

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
        content = f"({content})"

    # Get content within the outermost parens and strip whitespace
    inner_content = content[1:-1].strip()

    if not inner_content:  # Handles "()" or "( )"
        return "()"

    parts = inner_content.lower().split()  # Split into predicate name and args, convert to lowercase

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
            goal_predicate_tuples.add((op_name,) + args_lower)

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
    args = parser.parse_args()

    print(f"    INFO: Processing {args.pddl_problem_file} with {args.val_output_file}")

    try:
        # 1. Generate the master list of all possible predicates (normalized)
        ordered_master_pred_list = get_ordered_predicate_list(args.num_blocks, block_names_prefix="b")
        if not ordered_master_pred_list:
            print(f"ERROR: Could not generate master predicate list for {args.num_blocks} blocks.")
            exit(1)

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

        # 4. Convert each state in the trajectory to a binary vector
        binary_trajectory = []
        for state_preds_list_for_one_step in state_trajectory_pred_strings_list:
            state_preds_set_for_one_step = set(state_preds_list_for_one_step)  # Convert list to set
            binary_vector = state_preds_to_binary_vector(state_preds_set_for_one_step, ordered_master_pred_list)
            binary_trajectory.append(binary_vector)
        binary_trajectory_np = np.array(binary_trajectory, dtype=np.int8)

        # 5. Convert goal state to binary vector
        goal_binary_vector_np = state_preds_to_binary_vector(goal_state_preds_normalized_set, ordered_master_pred_list)

        # 6. Save outputs
        os.makedirs(os.path.dirname(args.text_trajectory_output), exist_ok=True)
        with open(args.text_trajectory_output, "w") as f_text:
            for i, state_preds_list in enumerate(state_trajectory_pred_strings_list):
                f_text.write(f"State {i}:\n")
                for pred in state_preds_list:  # Already sorted
                    f_text.write(f"  {pred}\n")
                f_text.write("\n")
            f_text.write("Goal Predicates (Normalized):\n")
            for pred in sorted(list(goal_state_preds_normalized_set)):
                f_text.write(f"  {pred}\n")

        os.makedirs(os.path.dirname(args.binary_trajectory_output), exist_ok=True)
        np.save(args.binary_trajectory_output, binary_trajectory_np)

        os.makedirs(os.path.dirname(args.binary_goal_output), exist_ok=True)
        np.save(args.binary_goal_output, goal_binary_vector_np)

        traj_len = (
            binary_trajectory_np.shape[0] if binary_trajectory_np.ndim > 0 and binary_trajectory_np.size > 0 else 0
        )
        vec_dim = (
            binary_trajectory_np.shape[1]
            if binary_trajectory_np.ndim > 1 and traj_len > 0
            else (len(ordered_master_pred_list) if ordered_master_pred_list else "N/A")
        )

        print(f"    INFO: Successfully processed. Trajectory Length: {traj_len}. Vector dim: {vec_dim}")
        print(f"    INFO: Text trajectory saved to: {args.text_trajectory_output}")
        print(f"    INFO: Binary trajectory saved to: {args.binary_trajectory_output}")
        print(f"    INFO: Binary goal saved to: {args.binary_goal_output}")

    except Exception as e:
        print(f"FATAL ERROR in parse_and_encode.py: {e}")
        import traceback

        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
