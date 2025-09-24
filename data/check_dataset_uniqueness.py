import argparse
import hashlib
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple


def parse_text_trajectory_file(filepath: Path) -> Tuple[Optional[str], Optional[str]]:
    """
    Parses a .traj.txt file to extract the initial state and goal state strings.
    Assumes initial state is the first line and goal state is marked by "Goal Predicates:".
    """
    initial_state_str: Optional[str] = None
    goal_state_str: Optional[str] = None

    try:
        with open(filepath, "r") as f:
            lines = f.readlines()

            if not lines:
                return None, None  # Empty file

            # Initial state is the first line
            initial_state_str = lines[0].strip()

            # Goal state is usually at the end, marked with "Goal Predicates:"
            for line in reversed(lines):  # Search from end for efficiency
                if "Goal Predicates:" in line:
                    match = re.search(r"Goal Predicates:\s*(.*)", line)
                    if match:
                        goal_state_str = match.group(1).strip()
                        break

            if not initial_state_str:
                print(f"Warning: Initial state not found in {filepath.name}.")
                return None, None
            if not goal_state_str:
                print(f"Warning: Goal predicates not found in {filepath.name}.")
                return None, None

    except FileNotFoundError:
        print(f"Error: File not found {filepath.name}")
        return None, None
    except Exception as e:
        print(f"Error parsing {filepath.name}: {e}")
        return None, None

    return initial_state_str, goal_state_str


def check_uniqueness(raw_block_dir: Path, num_blocks: int):
    """
    Checks the uniqueness of (initial state, goal state) pairs in the dataset.
    """
    print(f"Checking Dataset Uniqueness for N={num_blocks}")
    print(f"Scanning directory: {raw_block_dir}")

    traj_text_dir = raw_block_dir / "trajectories_text"
    if not traj_text_dir.is_dir():
        print(f"Error: 'trajectories_text' directory not found at {traj_text_dir}")
        print("Please ensure the dataset generation script has been run successfully.")
        return

    all_problems_count = 0
    unique_problem_hashes = set()
    # Stores hash -> list of basenames that share this hash
    duplicate_groups = defaultdict(list)

    for traj_file in traj_text_dir.glob("*.traj.txt"):
        all_problems_count += 1
        basename = traj_file.stem.replace(".traj", "")  # e.g., "blocks_4_problem_1"

        initial_str, goal_str = parse_text_trajectory_file(traj_file)

        if initial_str is None or goal_str is None:
            print(f"Skipping problem {basename} due to parsing errors in {traj_file.name}.")
            continue

        problem_hash = hashlib.sha256((initial_str + goal_str).encode("utf-8")).hexdigest()

        duplicate_groups[problem_hash].append(basename)
        unique_problem_hashes.add(problem_hash)

    print(f"\nUniqueness Report (N={num_blocks})")
    print(f"Total problems processed: {all_problems_count}")
    print(f"Total unique (initial, goal) pairs: {len(unique_problem_hashes)}")

    num_duplicate_instances = 0
    actual_duplicate_groups = {h: names for h, names in duplicate_groups.items() if len(names) > 1}

    if actual_duplicate_groups:
        print(f"\nFound {len(actual_duplicate_groups)} unique (initial, goal) pairs with duplicates:")
        for i, (problem_hash, basenames) in enumerate(actual_duplicate_groups.items()):
            num_duplicate_instances += len(basenames) - 1  # Count extra instances beyond the first
            print(f"  {i + 1}. Hash: {problem_hash[:10]}... (shared by {len(basenames)} problems)")
            print(f"     Problems: {', '.join(sorted(basenames))}")
    else:
        print("\nNo duplicate (initial, goal) pairs found!")

    print(f"\nTotal duplicate problem instances (beyond the first unique one): {num_duplicate_instances}")

    uniqueness_rate = (len(unique_problem_hashes) / all_problems_count) * 100 if all_problems_count > 0 else 0.0
    print(f"Uniqueness Rate (unique pairs / total problems): {uniqueness_rate:.2f}%")

    if uniqueness_rate < 100.0:
        print("\nNote: The dataset contains duplicate (initial, goal) pairs. The `analyze_dataset_splits.py` script")
        print("has been updated to ensure these duplicates are kept within the same train/val/test split,")
        print("preventing data leakage across splits for identical problems.")

    print("\nEnd of Report")


def main():
    parser = argparse.ArgumentParser(description="Check uniqueness of (initial state, goal state) pairs in a PaTS dataset.")
    parser.add_argument(
        "raw_block_dir",
        type=Path,
        help="Path to the raw problem data directory for a specific N (e.g., data/raw_problems/blocksworld/N4).",
    )
    parser.add_argument(
        "--num_blocks",
        type=int,
        required=True,
        help="Number of blocks for the dataset being checked (e.g., 4).",
    )

    args = parser.parse_args()

    if not args.raw_block_dir.is_dir():
        print(f"Error: Directory not found: {args.raw_block_dir}")
        return

    check_uniqueness(args.raw_block_dir, args.num_blocks)


if __name__ == "__main__":
    main()
