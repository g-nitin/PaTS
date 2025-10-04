import argparse
import hashlib
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator  # For better tick control
from scipy.stats import gaussian_kde


def get_plan_length(plan_filepath):
    """
    Counts the number of actions in a Fast Downward plan file.
    Assumes one action per line, and ignores lines starting with ';' (comments)
    or empty lines.
    """
    if not os.path.exists(plan_filepath) or os.path.getsize(plan_filepath) == 0:
        return 0  # Or raise an error, or return None
    count = 0
    try:
        with open(plan_filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith(";"):
                    count += 1
    except Exception as e:
        print(f"Warning: Could not read or process plan file {plan_filepath}: {e}")
        return 0  # Treat as 0 length if error
    return count


def analyze_and_split_plans(dataset_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=13, plot_kde=False):
    """
    Analyzes plan length distribution and creates stratified train/val/test splits.
    Generates a combined plot for train, val, test distributions.
    """
    random.seed(seed)

    np.random.seed(seed)  # For numpy operations like shuffling

    plans_dir = dataset_dir / "plans"
    trajectories_text_dir = dataset_dir / "trajectories_text"  # For initial/goal state parsing
    splits_output_dir = dataset_dir / "splits"  # New output directory for splits

    if not plans_dir.is_dir() or not trajectories_text_dir.is_dir():
        print(f"Error: Required directories not found at {plans_dir} or {trajectories_text_dir}")
        return

    # Ensure the splits output directory exists
    splits_output_dir.mkdir(parents=True, exist_ok=True)

    plan_files_data = []  # List of dicts (basename, plan_length, unique_problem_id)
    unique_problem_ids = set()  # To count unique (initial, goal) pairs

    for filename in os.listdir(plans_dir):
        if filename.endswith(".plan"):
            problem_basename = filename.replace(".plan", "")
            plan_filepath = plans_dir / filename
            length = get_plan_length(plan_filepath)

            # Load initial and goal states to create a unique problem ID
            if length > 0:  # Only consider problems with valid plans
                try:
                    # We need to load the initial and goal states to create a unique ID.
                    # For analyze_dataset_splits, we can use the text trajectories for hashing,
                    # as the binary ones might not exist yet for all encoding types.
                    # The text trajectory file contains initial state predicates and goal predicates.
                    text_traj_path = trajectories_text_dir / f"{problem_basename}.traj.txt"
                    if not text_traj_path.exists():
                        print(f"Warning: Text trajectory file for {problem_basename} not found. Skipping.")
                        continue

                    initial_state_str = ""
                    goal_state_str = ""
                    with open(text_traj_path, "r") as f:
                        lines = f.readlines()
                        # Initial state is the first state in the trajectory
                        # Goal state is usually at the end, marked with "Goal Predicates:"
                        if len(lines) > 0:
                            initial_state_str = lines[0].strip()
                        for line in lines:
                            if "Goal Predicates:" in line:
                                goal_state_str = line.split("Goal Predicates:")[1].strip()
                                break

                    # Create a unique identifier for the (initial, goal) pair
                    # Hash the string representations for a stable, unique ID
                    initial_goal_hash = hashlib.sha256((initial_state_str + goal_state_str).encode("utf-8")).hexdigest()

                    plan_files_data.append(
                        {"basename": problem_basename, "length": length, "unique_problem_id": initial_goal_hash}
                    )
                    unique_problem_ids.add(initial_goal_hash)

                except Exception as e:
                    print(f"Warning: Could not load state data for {problem_basename}: {e}. Skipping.")

            else:
                print(f"Warning: Plan for {problem_basename} has length 0 or could not be read. Skipping.")

    if not plan_files_data:
        print("Error: No valid plan files found to process.")
        return

    df = pd.DataFrame(plan_files_data)

    # Calculate and save max plan length
    if not df.empty:
        max_plan_length = int(df["length"].max())
        max_plan_length_path = splits_output_dir / "max_plan_length.txt"
        with open(max_plan_length_path, "w") as f:
            f.write(str(max_plan_length))
        print(f"Max plan length ({max_plan_length}) saved to: {max_plan_length_path}")
    else:
        print("Warning: DataFrame is empty, cannot determine max plan length.")

    splits_output_dir.mkdir(parents=True, exist_ok=True)  # Ensure splits output directory exists

    # ** 1. Analyze Overall Plan Length Distribution **
    print("\n** Overall Plan Length Distribution Analysis **")
    print(df["length"].describe())

    plt.figure(figsize=(10, 6))
    if not df.empty and df["length"].nunique() > 0:
        min_len_overall = df["length"].min()
        max_len_overall = df["length"].max()
        overall_bins = np.arange(min_len_overall - 0.5, max_len_overall + 1.5, 1)
        if len(overall_bins) <= 1:  # Fallback for single unique value
            overall_bins = np.array([min_len_overall - 0.5, min_len_overall + 0.5])

        plt.hist(
            df["length"],
            bins=overall_bins,  # type: ignore
            edgecolor="black",
            alpha=0.7,
            color="skyblue",
            density=False,
            label="Overall Counts",
        )
        plt.title("Overall Distribution of Plan Lengths")
        plt.xlabel("Plan Length")
        plt.ylabel("Number of Problems")
        plt.grid(axis="y", alpha=0.75)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, nbins="auto"))

        if plot_kde:
            ax_hist = plt.gca()
            if df["length"].nunique() > 1:  # KDE needs variation
                ax_kde = ax_hist.twinx()
                sns.kdeplot(df["length"], color="navy", linewidth=2, label="Overall KDE (density)", ax=ax_kde)  # type: ignore
                ax_kde.set_ylabel("Density")
                lines, labels = ax_hist.get_legend_handles_labels()
                lines2, labels2 = ax_kde.get_legend_handles_labels()
                ax_kde.legend(lines + lines2, labels + labels2, loc="upper right")
                ax_hist.legend().set_visible(False)
            else:
                ax_hist.legend(loc="upper right")  # Only hist legend
        else:
            plt.legend(loc="upper right")
    else:
        plt.text(0.5, 0.5, "No data to plot for overall distribution.", ha="center", va="center")

    distribution_plot_path = splits_output_dir / "plan_length_distribution_overall.png"
    plt.savefig(distribution_plot_path, dpi=150, bbox_inches="tight")
    print(f"Overall plan length distribution plot saved to: {distribution_plot_path}")
    plt.close()

    # ** 2. Stratified Splitting **
    print("\n** Generating Stratified Splits **")

    # Normalize ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0) and total_ratio > 0:
        print(f"Warning: Ratios do not sum to 1 (sum={total_ratio}). Normalizing.")
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio
        print(f"Normalized ratios: Train={train_ratio:.2f}, Val={val_ratio:.2f}, Test={test_ratio:.2f}")
    elif total_ratio == 0:
        print("Error: All ratios are zero. Cannot split.")
        return

    # Group by unique_problem_id first
    grouped_by_unique_id = df.groupby("unique_problem_id")

    # Collect all unique problem IDs
    all_unique_ids = list(grouped_by_unique_id.groups.keys())
    random.shuffle(all_unique_ids)  # Shuffle unique IDs before splitting

    # Determine split sizes for unique IDs
    n_total_unique = len(all_unique_ids)
    n_train_unique = int(np.floor(train_ratio * n_total_unique))
    n_val_unique = int(np.floor(val_ratio * n_total_unique))

    train_unique_ids = all_unique_ids[:n_train_unique]
    val_unique_ids = all_unique_ids[n_train_unique : n_train_unique + n_val_unique]
    test_unique_ids = all_unique_ids[n_train_unique + n_val_unique :]

    train_basenames, val_basenames, test_basenames = [], [], []

    # Populate basenames for each split
    for unique_id in train_unique_ids:
        train_basenames.extend(grouped_by_unique_id.get_group(unique_id)["basename"].tolist())
    for unique_id in val_unique_ids:
        val_basenames.extend(grouped_by_unique_id.get_group(unique_id)["basename"].tolist())
    for unique_id in test_unique_ids:
        test_basenames.extend(grouped_by_unique_id.get_group(unique_id)["basename"].tolist())

    # Shuffle the final lists as well, as items were added group by group
    random.shuffle(train_basenames)
    random.shuffle(val_basenames)
    random.shuffle(test_basenames)

    print(f"\nTotal unique (initial, goal) problem instances: {len(unique_problem_ids)}")
    print(f"\nTotal problems processed: {len(df)}")
    print(f"Train set size: {len(train_basenames)}")
    print(f"Validation set size: {len(val_basenames)}")
    print(f"Test set size: {len(test_basenames)}")

    # ** 3. Save Split Files **
    for split_name, basenames_list in [("train", train_basenames), ("val", val_basenames), ("test", test_basenames)]:
        filepath = splits_output_dir / f"{split_name}_files.txt"
        with open(filepath, "w") as f:
            for basename in basenames_list:
                f.write(basename + "\n")
        print(f"{split_name.capitalize()} basenames saved to: {filepath}")

    # ** 4. Verify Split Distribution and Plot Combined Overlaid Histogram **
    print("\n** Verifying Split Distributions & Plotting Combined Overlaid Histogram **")
    plt.figure(figsize=(16, 9))

    # Define colors and alpha for overlaid plot
    # Plotting order: train (most opaque), then val, then test (most transparent)
    plot_config = {
        "train": {"color": "#3675A3", "alpha": 0.8, "label": "Train"},  # Darker Blue
        "val": {"color": "#FF8C00", "alpha": 0.6, "label": "Val"},  # Orange
        "test": {"color": "#FFA07A", "alpha": 0.4, "label": "Test"},  # Light Salmon
    }
    plot_order = ["train", "val", "test"]  # Defines plotting order

    split_data_lengths = {}

    for split_name, basenames_list in [("train", train_basenames), ("val", val_basenames), ("test", test_basenames)]:
        if not basenames_list:
            split_data_lengths[split_name] = pd.Series(dtype="int64")  # Use int64 for consistency
            print(f"{split_name.capitalize()} set is empty.")
            continue
        split_df = df[df["basename"].isin(basenames_list)]
        split_data_lengths[split_name] = split_df["length"]
        print(f"\n{split_name.capitalize()} Set Plan Length Distribution (N={len(split_data_lengths[split_name])}):")
        if not split_data_lengths[split_name].empty:
            print(split_data_lengths[split_name].describe())

    all_lengths_list = [s for s in split_data_lengths.values() if not s.empty]
    if not all_lengths_list:
        print("No data in any split to plot. Skipping combined histogram.")
        plt.text(0.5, 0.5, "No data in any split to plot.", ha="center", va="center")
        plt.savefig(  # Use splits_output_dir
            splits_output_dir / "plan_length_distribution_splits_combined.png", dpi=150, bbox_inches="tight"
        )
        plt.close()
        return

    all_lengths_combined = pd.concat(all_lengths_list)
    if all_lengths_combined.empty:  # Should be caught by previous check, but good for safety
        print("Combined data is empty. Skipping combined histogram.")
        plt.text(0.5, 0.5, "No data in any split to plot.", ha="center", va="center")
        plt.savefig(  # Use splits_output_dir
            splits_output_dir / "plan_length_distribution_splits_combined.png", dpi=150, bbox_inches="tight"
        )
        plt.close()
        return

    min_overall_len_splits = int(all_lengths_combined.min())  # type: ignore
    max_overall_len_splits = int(all_lengths_combined.max())  # type: ignore

    bins = np.arange(min_overall_len_splits - 0.5, max_overall_len_splits + 1.5, 1)
    if len(bins) < 2:  # Handles cases like single unique value
        bins = np.array([min_overall_len_splits - 0.5, min_overall_len_splits + 0.5])

    print(f"Using bins for combined histogram: {bins}")

    for split_name in plot_order:
        if split_name in split_data_lengths and not split_data_lengths[split_name].empty:
            data_to_plot = split_data_lengths[split_name]
            config = plot_config[split_name]

            plt.hist(
                data_to_plot,
                bins=bins,  # type: ignore
                alpha=config["alpha"],
                label=f"{config['label']} (N={len(data_to_plot)})",
                color=config["color"],
                density=False,  # Plot raw counts
                edgecolor="none",  # No edges for cleaner overlay
                linewidth=0,
            )

    if plot_kde:
        if len(bins) > 1:
            bin_width = bins[1] - bins[0]
        else:  # Fallback if bins is not an array or has 1 element
            bin_width = 1.0  # Default bin width for scaling KDE if bins are ill-defined

        x_range_min = min_overall_len_splits
        x_range_max = max_overall_len_splits
        if x_range_min == x_range_max:  # Single point data
            x_range = np.array([x_range_min])
        else:
            x_range = np.linspace(x_range_min, x_range_max, 300)

        for split_name in plot_order:
            if split_name in split_data_lengths and not split_data_lengths[split_name].empty:
                data_series = split_data_lengths[split_name]
                config = plot_config[split_name]
                if len(data_series) > 1 and data_series.nunique() > 1:
                    try:
                        kde = gaussian_kde(data_series)
                        kde_values = kde(x_range)
                        kde_scale_factor = len(data_series) * bin_width
                        kde_values_scaled = kde_values * kde_scale_factor
                        plt.plot(
                            x_range,
                            kde_values_scaled,
                            color=config["color"],
                            linewidth=2,
                            linestyle="--",
                            alpha=0.9,  # Dashed line for KDE
                        )
                    except Exception as e:
                        print(f"Warning: Could not plot KDE for {split_name}: {e}")
                elif len(data_series) > 0:
                    print(f"Skipping KDE for {split_name} due to insufficient data or no variation.")

    plt.title("Plan Length Distribution by Set (Stratified Split)", fontsize=14, fontweight="bold")
    plt.xlabel("Plan Length", fontsize=12)
    plt.ylabel("Number of Problems", fontsize=12)  # Y-axis is counts
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.grid(axis="y", alpha=0.3, linestyle="--")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, nbins="auto"))  # Ensure integer ticks

    plt.tight_layout()

    combined_plot_path = splits_output_dir / "plan_length_distribution_splits_combined.png"
    plt.savefig(combined_plot_path, dpi=150, bbox_inches="tight")
    print(f"\nCombined distribution plot for splits saved to: {combined_plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze plan length distribution and create stratified train/val/test splits."
    )
    parser.add_argument(
        "raw_block_dir",
        type=Path,
        help="Path to the raw problem data directory for a specific N (e.g., data/raw_problems/blocksworld/N4).",
    )
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Proportion of data for training. Default is 0.7.")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Proportion of data for validation. Default is 0.15.")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Proportion of data for testing. Default is 0.15.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for reproducibility. Default is 13.")
    parser.add_argument(
        "--plot_kde",
        action="store_true",
        default=True,
        help="Include Kernel Density Estimate lines on the distribution plots. Default is True.",
    )

    args = parser.parse_args()

    if not (0 <= args.train_ratio <= 1 and 0 <= args.val_ratio <= 1 and 0 <= args.test_ratio <= 1):
        print("Error: Ratios must be between 0 and 1 (inclusive).")
        return

    current_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if np.isclose(current_sum, 0.0) and not (args.train_ratio > 0 or args.val_ratio > 0 or args.test_ratio > 0):
        print("Error: All ratios are zero. At least one ratio must be positive.")
        return
    # Normalization if sum is not 1.0 is handled in analyze_and_split_plans

    analyze_and_split_plans(args.raw_block_dir, args.train_ratio, args.val_ratio, args.test_ratio, args.seed, args.plot_kde)


if __name__ == "__main__":
    main()
