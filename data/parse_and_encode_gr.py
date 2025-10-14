import argparse
import json
import re
from pathlib import Path

import numpy as np
import pddlpy  # For parsing PDDL files
from pddlpy.pddl import Atom  # Import the Atom class


# Predicate List Generation
def get_ordered_predicate_list(num_robots, num_rooms, num_objects):
    """
    Generates a fixed, ordered list of all possible predicates for the Grippers domain.
    """
    robots = [f"robot{i + 1}" for i in range(num_robots)]
    rooms = [f"room{i + 1}" for i in range(num_rooms)]
    objects = [f"ball{i + 1}" for i in range(num_objects)]
    # Generate actual gripper object names for each robot
    grippers = []
    for i in range(num_robots):
        grippers.append(f"lgripper{i + 1}")
        grippers.append(f"rgripper{i + 1}")
    predicates = []

    # (at-robby ?r - robot ?x - room)
    for r_name in robots:
        for rm_name in rooms:
            predicates.append(f"(at-robby {r_name} {rm_name})")

    # (at ?o - object ?x - room)
    for o_name in objects:
        for rm_name in rooms:
            predicates.append(f"(at {o_name} {rm_name})")

    # (free ?r - robot ?g - gripper)
    for r_name in robots:
        for g_name in grippers:
            predicates.append(f"(free {r_name} {g_name})")

    # (carry ?r - robot ?o - object ?g - gripper)
    for r_name in robots:
        for o_name in objects:
            for g_name in grippers:
                predicates.append(f"(carry {r_name} {o_name} {g_name})")

    return predicates


def state_preds_to_sas_vector(
    state_predicates_set, num_robots, num_objects, robot_names, object_names, room_names, gripper_map
):
    """
    Converts a set of true predicate strings to a Grippers SAS+ position vector.
    Vector size: num_robots + num_objects
    - Part 1 (Robot Positions): vector[0:num_robots]
      - vector[i] = k, where robot i+1 is in room k (1-based).
    - Part 2 (Object Positions): vector[num_robots:]
      - vector[num_robots + j] = k > 0, where object j+1 is in room k.
      - vector[num_robots + j] = v < 0, where object j+1 is held by gripper with ID v.
    """
    vector_size = num_robots + num_objects
    position_vector = np.full(vector_size, -99, dtype=np.int8)  # Use -99 for unassigned

    # Create name-to-index maps for efficient lookup
    robot_to_idx = {name: i for i, name in enumerate(robot_names)}
    object_to_idx = {name: i for i, name in enumerate(object_names)}
    room_to_idx = {name: i + 1 for i, name in enumerate(room_names)}  # 1-based for SAS+ values

    for pred in state_predicates_set:
        # (at-robby rX roomY)
        m_at_robby = re.fullmatch(r"\(at-robby (r\w+\d+) (room\d+)\)", pred)
        if m_at_robby:
            robot_name, room_name = m_at_robby.groups()
            robot_idx = robot_to_idx[robot_name]
            room_val = room_to_idx[room_name]
            position_vector[robot_idx] = room_val
            continue

        # (at ballX roomY)
        m_at_obj = re.fullmatch(r"\(at (ball\d+) (room\d+)\)", pred)
        if m_at_obj:
            obj_name, room_name = m_at_obj.groups()
            obj_idx = object_to_idx[obj_name]
            room_val = room_to_idx[room_name]
            position_vector[num_robots + obj_idx] = room_val
            continue

        # (carry rX ballY gripperZ)
        m_carry = re.fullmatch(r"\(carry (robot\d+) (ball\d+) (lgripper\d+|rgripper\d+)\)", pred)
        if m_carry:
            robot_name, obj_name, gripper_name = m_carry.groups()
            obj_idx = object_to_idx[obj_name]
            gripper_val = gripper_map[gripper_name]
            position_vector[num_robots + obj_idx] = gripper_val
            continue

    if np.any(position_vector == -99):
        unassigned_indices = np.where(position_vector == -99)[0]
        error_messages = []
        for idx in unassigned_indices:
            if idx < num_robots:
                entity_name = robot_names[idx]
                entity_type = "robot"
            else:
                entity_name = object_names[idx - num_robots]
                entity_type = "object"
            error_messages.append(f"  - {entity_type} '{entity_name}' (index {idx})")

        raise ValueError(
            "Failed to assign a valid SAS+ position to all entities. Unassigned entities:\n"
            + "\n".join(error_messages)
            + f"\nPredicate set being processed: {state_predicates_set}"
        )

    return position_vector


# Predicate String Normalization
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


# PDDLPY Helper Function
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


# PDDLPY Helper Function
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


# VAL Output Parser
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


# State to Binary Vector
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

    parser.add_argument("--num-robots", required=True, type=int, help="Number of robots.")
    parser.add_argument("--num-rooms", required=True, type=int, help="Number of rooms.")
    parser.add_argument("--num-objects", required=True, type=int, help="Number of objects (balls).")

    parser.add_argument(
        "--encoding_type",
        type=str,
        choices=["bin", "sas"],
        default="bin",
        help="The type of state encoding to use: 'bin' (predicate onehot vector) or 'sas' (position vector).",
    )
    parser.add_argument(
        "--text_trajectory_output",
        required=True,
        help="Path to save text-based predicate trajectory.",
    )
    parser.add_argument(
        "--binary_output_prefix",
        required=True,
        help="Prefix for saving binary encoded state trajectory and goal state (.npy).",
    )

    args = parser.parse_args()

    print(f"    INFO: Processing {args.pddl_problem_file} with {args.val_output_file}")
    print(f"    INFO: Using encoding type: {args.encoding_type}")

    try:
        # Generate entity names once
        robot_names = [f"robot{i + 1}" for i in range(args.num_robots)]
        room_names = [f"room{i + 1}" for i in range(args.num_rooms)]
        object_names = [f"ball{i + 1}" for i in range(args.num_objects)]
        processed_dir = Path(args.binary_output_prefix).parent
        gripper_map: dict = {}

        # Generate actual gripper object names for each robot
        all_gripper_objects = []
        for i in range(args.num_robots):
            all_gripper_objects.append(f"lgripper{i + 1}")
            all_gripper_objects.append(f"rgripper{i + 1}")

        # 1. Generate encoding-specific information
        processed_dir.mkdir(parents=True, exist_ok=True)  # Ensure destination exists
        if args.encoding_type == "bin":
            ordered_master_pred_list = get_ordered_predicate_list(args.num_robots, args.num_rooms, args.num_objects)
            if not ordered_master_pred_list:
                print("ERROR: Could not generate master predicate list for Grippers config.")
                exit(1)

            manifest_file_path = processed_dir / "predicate_manifest.txt"
            with open(manifest_file_path, "w") as f_manifest:
                for pred_item in ordered_master_pred_list:
                    f_manifest.write(f"{pred_item}\n")
            print(f"    INFO: Predicate manifest saved to: {manifest_file_path}")
            encoding_info = {
                "type": "bin",
                "manifest_file": "predicate_manifest.txt",
                "feature_dim": len(ordered_master_pred_list),
                "domain_config": {"robots": args.num_robots, "rooms": args.num_rooms, "objects": args.num_objects},
            }
        elif args.encoding_type == "sas":
            ordered_master_pred_list = None  # Not used for SAS
            # Create a consistent mapping from gripper object name to a unique negative ID
            gripper_map = {name: -(i + 1) for i, name in enumerate(all_gripper_objects)}
            encoding_info = {
                "type": "sas",
                "feature_dim": args.num_robots + args.num_objects,
                "domain_config": {"robots": args.num_robots, "rooms": args.num_rooms, "objects": args.num_objects},
                "entity_order": {"robots": robot_names, "objects": object_names},
                "gripper_map": gripper_map,
            }
        else:
            raise ValueError(f"Unknown encoding type: {args.encoding_type}")

        # Save the encoding info file to the processed directory
        encoding_info_path = processed_dir / "encoding_info.json"
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
        if not initial_state_preds_normalized_set:
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
            if args.encoding_type == "bin":
                vector = state_preds_to_binary_vector(state_preds_set_for_one_step, ordered_master_pred_list)
            else:  # sas
                vector = state_preds_to_sas_vector(
                    state_preds_set_for_one_step,
                    args.num_robots,
                    args.num_objects,
                    robot_names,
                    object_names,
                    room_names,
                    gripper_map,
                )
            trajectory_vectors.append(vector)
        trajectory_np = np.array(trajectory_vectors, dtype=np.int8)

        # 5. Convert goal state to vector
        if args.encoding_type == "bin":
            goal_vector_np = state_preds_to_binary_vector(goal_state_preds_normalized_set, ordered_master_pred_list)
        else:  # sas
            # For SAS, the goal must be a complete state. The final state of the
            # expert trajectory is the complete goal state that satisfies the
            # (potentially partial) PDDL goal.
            goal_vector_np = trajectory_np[-1]

        # 6. Save text trajectory
        # The directory for text_trajectory_output should already be created by generate_dataset.sh
        with open(args.text_trajectory_output, "w") as f:
            for state_preds_list in state_trajectory_pred_strings_list:
                f.write(f"{' '.join(state_preds_list)}\n")
            # Also write the goal predicates for analyze_dataset_splits.py to use
            f.write("Goal Predicates: " + " ".join(sorted(list(goal_state_preds_normalized_set))) + "\n")

        # 7. Save binary outputs with new naming convention
        traj_output_path = Path(f"{args.binary_output_prefix}.traj.{args.encoding_type}.npy")
        goal_output_path = Path(f"{args.binary_output_prefix}.goal.{args.encoding_type}.npy")

        traj_output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure processed_trajectories dir exists
        np.save(goal_output_path, goal_vector_np)
        np.save(traj_output_path, trajectory_np)

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
