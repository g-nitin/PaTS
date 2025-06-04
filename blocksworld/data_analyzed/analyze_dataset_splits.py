import argparse
import os
import random

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


def analyze_and_split_plans(
    dataset_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=13, plot_kde=False
):
    """
    Analyzes plan length distribution and creates stratified train/val/test splits.
    Generates a combined plot for train, val, test distributions.
    """
    random.seed(seed)
    np.random.seed(seed)

    plans_dir = os.path.join(dataset_dir, "plans")
    if not os.path.isdir(plans_dir):
        print(f"Error: Plans directory not found at {plans_dir}")
        return

    plan_files_data = []  # List of tuples (basename, plan_length, full_plan_path)
    for filename in os.listdir(plans_dir):
        if filename.endswith(".plan"):
            problem_basename = filename.replace(".plan", "")
            plan_filepath = os.path.join(plans_dir, filename)
            length = get_plan_length(plan_filepath)
            if length > 0:  # Only consider problems with valid plans
                plan_files_data.append({"basename": problem_basename, "length": length, "path": plan_filepath})
            else:
                print(f"Warning: Plan for {problem_basename} has length 0 or could not be read. Skipping.")

    if not plan_files_data:
        print("Error: No valid plan files found to process.")
        return

    df = pd.DataFrame(plan_files_data)
    os.makedirs(output_dir, exist_ok=True)  # Ensure output_dir exists early

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
            bins=overall_bins,
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
                sns.kdeplot(df["length"], color="navy", linewidth=2, label="Overall KDE (density)", ax=ax_kde)
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

    distribution_plot_path = os.path.join(output_dir, "plan_length_distribution_overall.png")
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

    grouped_by_length = df.groupby("length")
    train_basenames, val_basenames, test_basenames = [], [], []

    for length, group_df in grouped_by_length:
        instances = group_df["basename"].tolist()
        random.shuffle(instances)
        n_total = len(instances)
        n_train = int(np.floor(train_ratio * n_total))
        n_val = int(np.floor(val_ratio * n_total))

        # Distribute remainder to ensure sum is n_total
        # Prioritize test set for its exact rounded share if possible, then adjust val
        n_test_ideal = int(np.round(test_ratio * n_total))
        current_alloc = n_train + n_val
        if current_alloc + n_test_ideal > n_total:
            n_test = n_total - current_alloc
            if n_test < 0:  # Should not happen if ratios sum to 1
                n_val = n_val + n_test  # reduce n_val
                n_test = 0
        else:
            n_test = n_total - current_alloc  # Default remainder to test

        # Remainder distribution
        remainder = n_total - (n_train + n_val + n_test)
        if remainder > 0:
            # Distribute remainder based on original proportions (simplistic: add to largest proportion first or test)
            if test_ratio >= train_ratio and test_ratio >= val_ratio:
                n_test += remainder
            elif train_ratio >= val_ratio:
                n_train += remainder
            else:
                n_val += remainder
        # Ensure no split is negative after adjustment
        n_train = max(0, n_train)
        n_val = max(0, n_val)
        n_test = max(0, n_total - n_train - n_val)

        train_basenames.extend(instances[:n_train])
        val_basenames.extend(instances[n_train : n_train + n_val])
        test_basenames.extend(instances[n_train + n_val : n_train + n_val + n_test])
        print(f"Length {length} (count {n_total}): Train={n_train}, Val={n_val}, Test={n_test}")

    # Shuffle the final lists as well, as items were added group by group
    random.shuffle(train_basenames)
    random.shuffle(val_basenames)
    random.shuffle(test_basenames)

    print(f"\nTotal problems processed: {len(df)}")
    print(f"Train set size: {len(train_basenames)}")
    print(f"Validation set size: {len(val_basenames)}")
    print(f"Test set size: {len(test_basenames)}")

    # ** 3. Save Split Files **
    for split_name, basenames_list in [("train", train_basenames), ("val", val_basenames), ("test", test_basenames)]:
        filepath = os.path.join(output_dir, f"{split_name}_files.txt")
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
        plt.savefig(
            os.path.join(output_dir, "plan_length_distribution_splits_combined.png"), dpi=150, bbox_inches="tight"
        )
        plt.close()
        return

    all_lengths_combined = pd.concat(all_lengths_list)
    if all_lengths_combined.empty:  # Should be caught by previous check, but good for safety
        print("Combined data is empty. Skipping combined histogram.")
        plt.text(0.5, 0.5, "No data in any split to plot.", ha="center", va="center")
        plt.savefig(
            os.path.join(output_dir, "plan_length_distribution_splits_combined.png"), dpi=150, bbox_inches="tight"
        )
        plt.close()
        return

    min_overall_len_splits = int(all_lengths_combined.min())
    max_overall_len_splits = int(all_lengths_combined.max())

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
                bins=bins,
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

    combined_plot_path = os.path.join(output_dir, "plan_length_distribution_splits_combined.png")
    plt.savefig(combined_plot_path, dpi=150, bbox_inches="tight")
    print(f"\nCombined distribution plot for splits saved to: {combined_plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze plan length distribution and create stratified train/val/test splits."
    )
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="Path to the root directory of the generated dataset (containing 'plans' subdirectory).",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to save the split files (train_files.txt, etc.) and distribution plots.",
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.7, help="Proportion of data for training. Default is 0.7."
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.15, help="Proportion of data for validation. Default is 0.15."
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.15, help="Proportion of data for testing. Default is 0.15."
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed for reproducibility. Default is 13.")
    parser.add_argument(
        "--plot_kde",
        action="store_true",
        help="Include Kernel Density Estimate lines on the distribution plots. Default is False.",
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

    analyze_and_split_plans(
        args.dataset_dir, args.output_dir, args.train_ratio, args.val_ratio, args.test_ratio, args.seed, args.plot_kde
    )


if __name__ == "__main__":
    main()
