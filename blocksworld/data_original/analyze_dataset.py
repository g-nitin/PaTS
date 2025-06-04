import argparse
import json
import os

import matplotlib.pyplot as plt
import seaborn as sns


def create_plan_length_distribution_plot(directory_path, output_image_name="plan_length_distribution.png"):
    """
    Reads all JSON files in a directory, extracts plan lengths,
    and creates a distribution plot with a KDE.

    Args:
        directory_path (str): The path to the directory containing JSON dataset files.
        output_image_name (str): The filename for the saved plot.
    """
    all_plan_lengths = []

    print(f"Scanning directory: {directory_path}")
    if not os.path.isdir(directory_path):
        print(f"Error: Directory '{directory_path}' not found.")
        return

    for filename in os.listdir(directory_path):
        if filename.endswith(".json") and filename.startswith("dataset_"):
            filepath = os.path.join(directory_path, filename)
            print(f"Processing file: {filepath}")
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)

                if "plans" in data and isinstance(data["plans"], list):
                    for plan_obj in data["plans"]:
                        if "actions" in plan_obj and isinstance(plan_obj["actions"], list):
                            all_plan_lengths.append(len(plan_obj["actions"]))
                        else:
                            print(f"  Warning: 'actions' key missing or not a list in a plan object in {filename}")
                else:
                    print(f"  Warning: 'plans' key missing or not a list in {filename}")

            except json.JSONDecodeError:
                print(f"  Error: Could not decode JSON from {filename}")
            except Exception as e:
                print(f"  An unexpected error occurred with {filename}: {e}")

    if not all_plan_lengths:
        print("No plan lengths found to plot.")
        return

    print(f"\nFound {len(all_plan_lengths)} plan lengths in total.")
    print(
        f"Min length: {min(all_plan_lengths)}, Max length: {max(all_plan_lengths)}, Avg length: {sum(all_plan_lengths) / len(all_plan_lengths):.2f}"
    )

    # Create the distribution plot
    plt.figure(figsize=(12, 7))

    # Using histplot for more direct control over histogram and KDE on the same axes
    # discrete=True is good for integer data like plan lengths
    # stat="density" normalizes histogram to match KDE scale
    sns.histplot(
        all_plan_lengths,
        kde=True,
        stat="density",
        discrete=True,
        label="Plan Lengths",
        color="skyblue",
        edgecolor="black",
    )

    # Alternatively, sns.displot is a figure-level function:
    # sns.displot(all_plan_lengths, kind="hist", kde=True)
    # This would create its own figure, so you'd call plt.title etc. after it.

    plt.title(f"Distribution of Plan Lengths (from directory: {os.path.basename(directory_path)})", fontsize=16)
    plt.xlabel("Plan Length (Number of Actions)", fontsize=14)
    plt.ylabel("Density", fontsize=14)

    # Improve x-axis ticks for integer values
    if all_plan_lengths:
        min_val = min(all_plan_lengths)
        max_val = max(all_plan_lengths)
        # Only set specific ticks if the range is manageable
        if max_val - min_val < 25:
            plt.xticks(range(min_val, max_val + 1))
        else:
            # For larger ranges, let matplotlib/seaborn decide, or use a locator
            pass

    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()  # Adjusts plot to prevent labels from being cut off

    # Save the plot
    plt.savefig(output_image_name)
    print(f"\nDistribution plot saved as '{output_image_name}'")

    # Show the plot
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a distribution plot of plan lengths from JSON datasets.")
    parser.add_argument("directory", type=str, help="Path to the directory containing JSON dataset files.")
    parser.add_argument("--output", type=str, default="plan_length_distribution.png", help="Output image file name.")
    args = parser.parse_args()
    create_plan_length_distribution_plot(args.directory, args.output)
