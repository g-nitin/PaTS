import os
import random
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from pddlpy import DomainProblem
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ** Configuration **
NUM_RUNS = 3  # Number of times to run the evaluation
BASE_SEED = 13  # Base seed for reproducibility

PLANNING_DIR = Path("/Users", "nitingupta", "usc", "ai4s", "libraries", "planning")

MODEL_PATH = PLANNING_DIR / "plansformer_v3" / "model_files"
DOMAIN_FILE_PATH = PLANNING_DIR / "blocksworld_domain.pddl"

VAL_EXECUTABLE_PATH = PLANNING_DIR / "VAL" / "validate"
FD_SCRIPT_PATH = PLANNING_DIR / "downward" / "fast-downward.py"
FD_ALIAS = "seq-opt-lmcut"  # Optimal sequential planner alias

ROOT_DIR = Path(__file__).parent.resolve().parent.parent
PROBLEM_FILES_DIR = ROOT_DIR / "data" / "blocks_4" / "pddl"
OUTPUT_DIR = ROOT_DIR / "outputs" / "plansformer"

# If OUTPUT_DIR exists, ask for confirmation to overwrite
if OUTPUT_DIR.exists():
    confirm = input(f"Output directory {OUTPUT_DIR} already exists. Overwrite? (y/n): ")
    if confirm.lower() != "y":  # User chose not to overwrite
        print("Exiting without overwriting.")
        exit(0)
    print(f"Overwriting existing output directory: {OUTPUT_DIR}")
    # Remove existing directory
    # Check if the directory is empty
    if not any(OUTPUT_DIR.iterdir()):
        # Delete the empty directory
        OUTPUT_DIR.rmdir()
        print(f"Directory '{OUTPUT_DIR}' deleted successfully.")
    else:
        # Delete the directory and all its contents
        shutil.rmtree(OUTPUT_DIR)
        print(f"Directory '{OUTPUT_DIR}' and its contents deleted successfully.")

# Create output directory if it doesn't exist
if not OUTPUT_DIR.exists():
    print(f"Creating output directory: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def format_atom_for_plansformer(atom_obj, for_action_schema=False):
    """Formats a pddlpy Atom object into a string for Plansformer.
    Atom.predicate is a list, e.g., ['on', 'b1', 'b2'] or ['on', '?x', '?y']
    """
    pred_name = atom_obj.predicate[0]
    # For action schemas, Plansformer examples use 'x', 'y' not '?x', '?y'
    if for_action_schema:
        pred_args = " ".join([arg.replace("?", "") for arg in atom_obj.predicate[1:]])
    else:
        pred_args = " ".join(atom_obj.predicate[1:])
    return f"{pred_name} {pred_args}".strip()


def format_pddl_action(action_name, action_op):
    """Formats an Operator object from pddlpy for Plansformer."""
    # Parameters: pddlpy's Operator.variable_list seems to be
    # for grounding. The listener's enterTypedVariableList populates it.
    # We'll assume it contains {'?x': type, '?y': type} for the schema.
    # Plansformer examples don't explicitly list parameters in the <ACTION name params> part,
    # but they are used in PRE and EFFECT.

    # Preconditions
    preconditions_str = []
    for atom_obj in action_op.precondition_pos:
        preconditions_str.append(format_atom_for_plansformer(atom_obj, for_action_schema=True))

    effects_str = []
    for atom_obj in action_op.effect_pos:
        effects_str.append(format_atom_for_plansformer(atom_obj, for_action_schema=True))
    for atom_obj in action_op.effect_neg:
        effects_str.append(f"not {format_atom_for_plansformer(atom_obj, for_action_schema=True)}")

    return f"<ACTION> {action_name} <PRE> {', '.join(preconditions_str)} <EFFECT> {', '.join(effects_str)}"


def pddl_to_plansformer_input(domain_file_path_str, problem_file_path_str):
    """
    Converts PDDL domain and problem files to Plansformer input string
    using the pddl structure.
    """
    try:
        # pddl.DomainProblem expects file paths
        dp = DomainProblem(domain_file_path_str, problem_file_path_str)
    except Exception as e:
        print(
            f"Error parsing PDDL files '{domain_file_path_str}' or '{problem_file_path_str}' with pddl.DomainProblem: {e}"
        )
        return None

    # Goal
    goal_atoms_str = [format_atom_for_plansformer(atom) for atom in dp.problem.goals]
    goal_str = ", ".join(goal_atoms_str)

    # Init
    init_atoms_str = [format_atom_for_plansformer(atom) for atom in dp.problem.initialstate]
    init_str = ", ".join(init_atoms_str)

    # Actions
    action_strs = []
    for action_name in dp.domain.operators.keys():  # This gets action schemas
        action_op = dp.domain.operators[action_name]
        action_strs.append(format_pddl_action(action_name, action_op))

    actions_full_str = " ".join(action_strs)

    # Construct the final string
    # Example: <GOAL> on b1 b2, clear b1, ontable b2 <INIT> handempty, ontable b1, clear b1, ontable b2, clear b2 <ACTION> pick-up <PRE> clear x, ontable x, handempty <EFFECT> not ontable x, not clear x, not handempty, holding x ...
    final_input_string = f"<GOAL> {goal_str} <INIT> {init_str} {actions_full_str}"
    return final_input_string


def generate_plan_with_plansformer(input_string, model, tokenizer, device):
    """Generates a plan using the Plansformer model."""
    start_time = time.time()
    inputs = tokenizer(input_string, return_tensors="pt", truncation=True, max_length=512).to(device)
    output_sequences = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=150,  # From Plansformer paper (Table 1, Max Target Text Length)
        num_beams=2,  # From Plansformer supplementary
        repetition_penalty=2.5,  # From Plansformer supplementary
        length_penalty=1.0,  # From Plansformer supplementary
        early_stopping=True,  # Good practice
    )
    generation_time = time.time() - start_time
    generated_plan_string = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    return generated_plan_string, generation_time


def parse_plansformer_output_to_val(plan_string):
    """Converts Plansformer's comma-separated plan to VAL format (list of actions)."""
    if not plan_string.strip():
        return []
    actions = plan_string.split(",")
    val_actions = []
    for action in actions:
        action = action.strip()
        if action:
            parts = action.split()
            val_actions.append(f"({parts[0]} {' '.join(parts[1:])})")
    return val_actions


def validate_plan_with_val(domain_file, problem_file, val_plan_actions):
    """Validates a plan using VAL and extracts metrics."""
    if not val_plan_actions:  # Empty plan
        return False, 0, "Empty plan generated"

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".plan") as tmp_plan_file:
        tmp_plan_file.write("\n".join(val_plan_actions))
        tmp_plan_file_path = tmp_plan_file.name
    is_valid, plan_length, error_message = False, 0, "VAL execution failed or plan invalid"
    try:
        # VAL command: validate <domain> <problem> <plan>
        process = subprocess.run(
            [VAL_EXECUTABLE_PATH, "-v", domain_file, problem_file, tmp_plan_file_path],
            capture_output=True,
            text=True,
            timeout=30,
        )

        val_output = process.stdout + "\n" + process.stderr
        # print(f"** VAL Output for {problem_file} **")
        # print(val_output)
        # print("** End VAL Output **")

        if (
            "Plan valid" in val_output or "Plan validated" in val_output or "Solution Found" in val_output
        ):  # VAL output can vary
            is_valid = True
            # Try to find plan length
            length_match = re.search(r"Plan length:\s*(\d+)", val_output, re.IGNORECASE)
            if not length_match:  # Another common VAL output format
                length_match = re.search(r"Final value:\s*(\d+)", val_output, re.IGNORECASE)  # IPC version
            if length_match:
                plan_length = int(length_match.group(1))
            else:  # If length not found but plan is valid, count actions
                plan_length = len(val_plan_actions)
            error_message = ""
        else:
            # Try to find specific error messages
            if "Goal not satisfied" in val_output:
                error_message = "Goal not satisfied"
            elif "Failed to apply action" in val_output:
                error_message = "Failed to apply action"
            elif "Syntax error" in val_output:
                error_message = "Plan syntax error"
            else:
                error_message = "Plan invalid (unknown reason)"
                # Capture first few lines of error if any
                err_lines = [
                    line for line in val_output.splitlines() if "error" in line.lower() or "failed" in line.lower()
                ]
                if err_lines:
                    error_message += ": " + err_lines[0]

    except subprocess.TimeoutExpired:
        error_message = "VAL timed out"
    except Exception as e:
        error_message = f"VAL execution error: {e}"
    finally:
        os.remove(tmp_plan_file_path)

    return is_valid, plan_length, error_message


def get_fast_downward_plan_stats(domain_file, problem_file, fd_cache):
    """Gets plan statistics from Fast Downward."""
    if problem_file in fd_cache:
        return fd_cache[problem_file]
    if not os.path.exists(FD_SCRIPT_PATH):
        # print(f"Warning: Fast Downward script not found at {FD_SCRIPT_PATH}. Skipping FD comparison.")
        fd_cache[problem_file] = (False, 0, [])
        return False, 0, []

    output_plan_file = "sas_plan"  # Default FD output
    if os.path.exists(output_plan_file):
        os.remove(output_plan_file)
    res = (False, 0, [])
    try:
        cmd = [str(FD_SCRIPT_PATH), "--alias", FD_ALIAS, str(domain_file), str(problem_file)]
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if os.path.exists(output_plan_file):
            with open(output_plan_file, "r") as f:
                plan_actions = [line.strip() for line in f if not line.startswith(";")]

            # Validate FD plan with VAL to be sure and get consistent length
            # This also ensures FD plan is in VAL format if we need the actions
            is_valid_val, length_val, _ = validate_plan_with_val(str(domain_file), str(problem_file), plan_actions)
            if is_valid_val:
                res = (True, length_val, plan_actions)
            else:
                print(f"Warning: FD plan for {problem_file} deemed invalid by VAL.")
                res = (False, 0, [])
        else:
            # Check FD output for "Solution found." or similar
            if (
                "Solution found." in process.stdout or "Search time" in process.stdout
            ):  # Heuristic: FD might have found a plan but writing failed or different name
                print(
                    f"Warning: FD found a solution for {problem_file} but {output_plan_file} not found. Output:\n{process.stdout}"
                )
            res = (False, 0, [])

    except subprocess.TimeoutExpired:
        print(f"Fast Downward timed out for {problem_file}")
    except Exception as e:
        print(f"Error running Fast Downward for {problem_file}: {e}")
    finally:
        if os.path.exists(output_plan_file):
            os.remove(output_plan_file)
        if os.path.exists("output.sas"):
            os.remove("output.sas")
    fd_cache[problem_file] = res
    return res


def set_seeds(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


# ** Main Evaluation Logic **
def main():
    print("Loading Plansformer model and tokenizer...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)
    model.eval()  # Set to evaluation mode
    print(f"Model loaded on {device}.")

    if not DOMAIN_FILE_PATH.exists():
        print(f"Domain file not found: {DOMAIN_FILE_PATH}")
        return

    problem_files_paths = list(PROBLEM_FILES_DIR.glob("*.pddl"))
    if not problem_files_paths:
        print(f"No PDDL problem files found in {PROBLEM_FILES_DIR}")
        return
    print(f"Found {len(problem_files_paths)} problem files.\n")

    # print(f"Using {(num := 0.5)}  for debugging")

    all_runs_results_list = []  # To store detailed results from all runs
    per_run_metrics_list = []  # To store summary metrics for each run

    fd_cache = {}  # Cache FD results as they are deterministic

    for run_idx in range(NUM_RUNS):
        current_seed = BASE_SEED + run_idx
        set_seeds(current_seed)
        print(f"\n** Starting Run {run_idx + 1}/{NUM_RUNS} (Seed: {current_seed}) **")

        run_detailed_results = []
        plansformer_valid_plans_count = 0
        plansformer_optimal_plans_count = 0
        total_plansformer_plan_length_for_run = 0
        # total_fd_plan_length_for_run = 0 # FD length is per problem, not summed per run directly for avg
        num_fd_valid_for_run = 0
        total_generation_time_for_run = 0
        valid_pf_plan_lengths_for_run = []
        valid_fd_plan_lengths_for_run = []  # For problems where FD was valid

        for i, prob_file_path in enumerate(problem_files_paths):
            prob_file_path_str = str(prob_file_path)  # Ensure it's a string for pddlpy
            print(f"  Run {run_idx + 1}, Problem {i + 1}/{len(problem_files_paths)}: {prob_file_path.name}", end="\r")

            plansformer_input_str = pddl_to_plansformer_input(str(DOMAIN_FILE_PATH), prob_file_path_str)
            if not plansformer_input_str:
                # Handle parsing error (already printed in the function)
                # Add a minimal entry to detailed results if needed
                run_detailed_results.append(
                    {
                        "run_id": run_idx + 1,
                        "seed": current_seed,
                        "problem": prob_file_path.name,
                        "plansformer_is_valid": False,
                        "plansformer_val_error": "PDDL parsing error for Plansformer input",
                    }
                )
                continue

            raw_plan_str, gen_time = generate_plan_with_plansformer(plansformer_input_str, model, tokenizer, device)
            total_generation_time_for_run += gen_time
            val_plan_actions = parse_plansformer_output_to_val(raw_plan_str)
            pf_is_valid, pf_plan_length, pf_val_error = validate_plan_with_val(
                str(DOMAIN_FILE_PATH), prob_file_path_str, val_plan_actions
            )

            if pf_is_valid:
                plansformer_valid_plans_count += 1
                total_plansformer_plan_length_for_run += pf_plan_length
                valid_pf_plan_lengths_for_run.append(pf_plan_length)

            fd_is_valid, fd_plan_length, _ = get_fast_downward_plan_stats(DOMAIN_FILE_PATH, prob_file_path, fd_cache)
            if fd_is_valid:
                # total_fd_plan_length_for_run += fd_plan_length # Not needed for avg of avgs
                num_fd_valid_for_run += 1
                valid_fd_plan_lengths_for_run.append(fd_plan_length)

            is_optimal = False
            if pf_is_valid and fd_is_valid and pf_plan_length == fd_plan_length:
                plansformer_optimal_plans_count += 1
                is_optimal = True

            problem_result_detail = {
                "run_id": run_idx + 1,
                "seed": current_seed,
                "problem": prob_file_path.name,
                "plansformer_input_snippet": plansformer_input_str[:200] + "...",
                "plansformer_raw_plan": raw_plan_str,
                "plansformer_gen_time": gen_time,
                "plansformer_val_actions": "; ".join(val_plan_actions),
                "plansformer_is_valid": pf_is_valid,
                "plansformer_plan_length": pf_plan_length if pf_is_valid else 0,
                "plansformer_val_error": pf_val_error,
                "fd_is_valid": fd_is_valid,
                "fd_plan_length": fd_plan_length if fd_is_valid else 0,
                "plansformer_is_optimal": is_optimal,
            }
            run_detailed_results.append(problem_result_detail)
            all_runs_results_list.append(problem_result_detail)  # Also add to global list for one big CSV

        print()  # Newline after problem progress
        # Calculate metrics for this run
        num_problems_this_run = len(problem_files_paths)
        run_coverage = (plansformer_valid_plans_count / num_problems_this_run) * 100 if num_problems_this_run > 0 else 0
        run_avg_pf_plan_length = np.mean(valid_pf_plan_lengths_for_run) if valid_pf_plan_lengths_for_run else 0
        run_optimality_rate = (
            (plansformer_optimal_plans_count / plansformer_valid_plans_count) * 100
            if plansformer_valid_plans_count > 0
            else 0
        )
        run_avg_gen_time = (total_generation_time_for_run / num_problems_this_run) if num_problems_this_run > 0 else 0

        per_run_metrics_list.append(
            {
                "run_id": run_idx + 1,
                "seed": current_seed,
                "coverage_percent": run_coverage,
                "avg_plansformer_plan_length": run_avg_pf_plan_length,
                "optimality_rate_percent": run_optimality_rate,
                "avg_generation_time_sec": run_avg_gen_time,
                "num_pf_valid": plansformer_valid_plans_count,
                "num_optimal": plansformer_optimal_plans_count,
                "num_problems": num_problems_this_run,
            }
        )
        print(
            f"  Run {run_idx + 1} Summary: Coverage: {run_coverage:.2f}%, Avg.Len: {run_avg_pf_plan_length:.2f}, Optimality: {run_optimality_rate:.2f}%"
        )

    # ** Aggregate and Print Overall Metrics **
    print("\n** Overall Evaluation Summary (Aggregated Across Runs) **")
    per_run_df = pd.DataFrame(per_run_metrics_list)
    per_run_df_path = str(OUTPUT_DIR / "plansformer_per_run_metrics.csv")
    per_run_df.to_csv(per_run_df_path, index=False)
    print(f"Per-run metrics saved to {per_run_df_path}")

    if not per_run_df.empty:
        mean_coverage = per_run_df["coverage_percent"].mean()
        std_coverage = per_run_df["coverage_percent"].std()
        print(f"Plansformer Avg. Coverage: {mean_coverage:.2f}% (Std: {std_coverage:.2f}%)")

        mean_avg_pf_len = per_run_df["avg_plansformer_plan_length"].mean()
        std_avg_pf_len = per_run_df["avg_plansformer_plan_length"].std()
        print(f"Plansformer Avg. Plan Length (for valid plans): {mean_avg_pf_len:.2f} (Std: {std_avg_pf_len:.2f})")

        mean_optimality = per_run_df["optimality_rate_percent"].mean()
        std_optimality = per_run_df["optimality_rate_percent"].std()
        print(f"Plansformer Avg. Optimality Rate: {mean_optimality:.2f}% (Std: {std_optimality:.2f}%)")

        mean_gen_time = per_run_df["avg_generation_time_sec"].mean()
        std_gen_time = per_run_df["avg_generation_time_sec"].std()
        print(f"Plansformer Avg. Generation Time: {mean_gen_time:.4f}s (Std: {std_gen_time:.4f}s)")
    else:
        print("No runs completed to calculate aggregate metrics.")

    # Save all detailed results to one CSV
    all_results_df = pd.DataFrame(all_runs_results_list)
    all_results_df_path = str(OUTPUT_DIR / "plansformer_all_runs_detailed_results.csv")
    all_results_df.to_csv(all_results_df_path, index=False)
    print(f"\nAll detailed results from all runs saved to {all_results_df_path}")

    # ** Visualization **
    if not per_run_df.empty:
        print("\nGenerating visualizations...")
        plt.style.use("seaborn-v0_8-whitegrid")  # Using an available style

        # Coverage per run
        plt.figure(figsize=(10, 6))
        sns.barplot(x="run_id", y="coverage_percent", data=per_run_df, hue="run_id", palette="viridis", legend=False)
        plt.title("Plansformer Coverage per Run")
        plt.xlabel("Run ID")
        plt.ylabel("Coverage (%)")
        plt.ylim(0, 100)
        plt.savefig(str(OUTPUT_DIR / "plansformer_coverage_per_run.png"))
        plt.close()

        # Optimality per run
        plt.figure(figsize=(10, 6))
        sns.barplot(x="run_id", y="optimality_rate_percent", data=per_run_df, hue="run_id", palette="mako", legend=False)
        plt.title("Plansformer Optimality Rate per Run")
        plt.xlabel("Run ID")
        plt.ylabel("Optimality Rate (%)")
        plt.ylim(0, 100)
        plt.savefig(str(OUTPUT_DIR / "plansformer_optimality_per_run.png"))
        plt.close()

        # Distribution of Plan Lengths (from the last run, or could aggregate all valid plans)
        # For simplicity, using the last run's detailed results for plan length distribution
        last_run_id = per_run_df["run_id"].max()
        last_run_details_df = all_results_df[all_results_df["run_id"] == last_run_id]

        valid_pf_plans_last_run = last_run_details_df[last_run_details_df["plansformer_is_valid"]]
        valid_fd_plans_last_run = last_run_details_df[
            last_run_details_df["fd_is_valid"]
        ]  # Assuming FD results are consistent

        if not valid_pf_plans_last_run.empty:
            plt.figure(figsize=(12, 7))
            sns.histplot(
                valid_pf_plans_last_run["plansformer_plan_length"],
                color="skyblue",
                label="Plansformer Valid Plan Lengths",
                kde=True,
                stat="density",
                common_norm=False,
            )
            if not valid_fd_plans_last_run.empty:
                sns.histplot(
                    valid_fd_plans_last_run["fd_plan_length"],
                    color="orange",
                    label="Fast Downward Plan Lengths",
                    kde=True,
                    stat="density",
                    common_norm=False,
                )
            plt.title(f"Distribution of Plan Lengths (Run {last_run_id})")
            plt.xlabel("Plan Length")
            plt.ylabel("Density")
            plt.legend()
            plt.savefig(str(OUTPUT_DIR / "plansformer_plan_length_distribution.png"))
            plt.close()

        # Distribution of Generation Times (from all runs)
        if not all_results_df.empty:
            plt.figure(figsize=(10, 6))
            sns.histplot(all_results_df["plansformer_gen_time"], color="lightcoral", kde=True)  # type: ignore
            plt.title("Distribution of Plansformer Generation Times (All Runs)")
            plt.xlabel("Generation Time (seconds)")
            plt.ylabel("Frequency")
            plt.savefig(str(OUTPUT_DIR / "plansformer_generation_time_distribution.png"))
            plt.close()

        print("Visualizations saved as PNG files.")


if __name__ == "__main__":
    if not DOMAIN_FILE_PATH.exists():
        print(f"Domain file '{DOMAIN_FILE_PATH}' doesn't exist. Exiting.")
        exit(1)
    if not PROBLEM_FILES_DIR.exists() or not any(PROBLEM_FILES_DIR.iterdir()):
        print(f"Problem files directory '{PROBLEM_FILES_DIR}' doesn't exist or is empty. Exiting.")
        exit(1)

    # https://stackoverflow.com/a/62703850
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    main()
