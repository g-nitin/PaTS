import argparse
import hashlib
from pathlib import Path

from check_dataset_uniqueness import parse_text_trajectory_file


def get_problem_hash(text_traj_file: Path) -> str:
    initial_str, goal_str = parse_text_trajectory_file(text_traj_file)
    if initial_str is None or goal_str is None:
        return ""  # Indicate failure
    return hashlib.sha256((initial_str + goal_str).encode("utf-8")).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get unique hash for a problem from its text trajectory file.")
    parser.add_argument("text_traj_file", type=Path, help="Path to the .traj.txt file.")
    args = parser.parse_args()

    problem_hash = get_problem_hash(args.text_traj_file)
    if problem_hash:
        print(problem_hash)
    else:
        # Exit with non-zero code to indicate error
        exit(1)
