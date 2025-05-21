import argparse
import json
import random
import sys
from pathlib import Path
from pprint import pformat

import torch
from BlocksWorld import BlocksWorldGenerator
from loguru import logger


# ** Logging Configuration **
def setup_logging(log_level: str):
    """Configures Loguru logger."""
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        level=log_level.upper(),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,  # Ensure logs are colorized in terminal
    )
    logger.info(f"Logging initialized at level {log_level.upper()}")


# ** Core Functionality **
def display_single_plan_details(generator: BlocksWorldGenerator, plan_data: dict):
    """
    Displays detailed information about a single generated plan using logger.
    """
    logger.info("** Single Plan Details **")
    logger.info(f"Number of states in plan: {len(plan_data['plan'])}")

    # Convert list-based states to tensors for decoding
    initial_state_tensor = torch.tensor(plan_data["initial_state"], dtype=torch.int)
    goal_state_tensor = torch.tensor(plan_data["goal_state"], dtype=torch.int)

    logger.info(f"Initial State (binary vector):\n{plan_data['initial_state']}")
    decoded_initial_map = generator.decode_vector(initial_state_tensor)
    # Log feature map at DEBUG level as it can be verbose
    logger.debug(f"Initial State (feature map):\n{pformat(decoded_initial_map)}")
    logger.info(f"Initial State (decoded):\n{pformat(generator.decode_state(decoded_initial_map))}")

    decoded_goal_map = generator.decode_vector(goal_state_tensor)
    logger.debug(f"Goal State (feature map):\n{pformat(decoded_goal_map)}")
    logger.info(f"Goal State (decoded):\n{pformat(generator.decode_state(decoded_goal_map))}")

    logger.info("Plan Actions:")
    # Log actions one by one for better readability if many
    for i, action in enumerate(plan_data["actions"]):
        logger.info(f"  Action {i + 1}: {action}")

    logger.info("Decoded Plan States (sequence):")
    for i, state_vector_list in enumerate(plan_data["plan"]):
        state_tensor = torch.tensor(state_vector_list, dtype=torch.int)
        decoded_state_map = generator.decode_vector(state_tensor)
        fully_decoded_state = generator.decode_state(decoded_state_map)
        logger.info(f"  State {i}: {pformat(fully_decoded_state)}")
        # Conditionally log raw vector if DEBUG level is active
        if logger.level("DEBUG").no >= logger.level("INFO").no:
            logger.debug(f"    Raw vector for State {i}: {state_vector_list}")

    logger.info("** End of Single Plan Details **")


def generate_and_save_dataset(
    generator: BlocksWorldGenerator, num_plans: int, output_file_path_str: str | None
):
    """
    Generates a dataset of plans and saves it to a JSON file if a path is provided.
    """
    logger.info(
        f"Starting dataset generation: {num_plans} plans for {generator.num_blocks} blocks."
    )

    # Note: BlocksWorldGenerator.generate_dataset prints progress using `end="\r"`.
    # This might interfere with loguru's output. For cleaner logs, this print
    # could be removed or made conditional within BlocksWorldGenerator.
    dataset = generator.generate_dataset(num_plans=num_plans)

    logger.success(f"Dataset generation complete. Generated {len(dataset['plans'])} plans.")
    logger.info(
        f"Each state in the dataset is encoded with {dataset['num_features']} binary features."
    )

    if output_file_path_str:
        output_path = Path(output_file_path_str)
        # Create parent directories if they don't exist
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured output directory exists: {output_path.parent}")
        except Exception as e:
            logger.warning(
                f"Could not create directory {output_path.parent}: {e}. Saving may fail if it doesn't exist."
            )

        logger.info(f"Saving dataset to {output_path}...")
        try:
            with open(output_path, "w") as f:
                json.dump(dataset, f, indent=2)  # Use indent for readable JSON
            logger.success(f"Dataset successfully saved to {output_path}")
        except IOError as e:
            logger.error(f"Failed to save dataset to {output_path}: {e}")
    else:
        logger.info("No output file specified. Dataset will be generated but not saved to disk.")
        if dataset["plans"] and logger.level("DEBUG").no >= logger.level("INFO").no:
            logger.debug(
                f"First plan in generated dataset (not saved):\n{pformat(dataset['plans'][0])}"
            )


def main():
    """
    Main function to parse arguments and orchestrate plan generation.
    """
    parser = argparse.ArgumentParser(
        description="Generate Blocks World plans. Can generate a single plan with details or a dataset of plans.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # Shows default values in help
    )

    parser.add_argument(
        "--num-blocks",
        type=int,
        required=True,
        help="Number of blocks in the Blocks World environment (e.g., 3).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "dataset"],
        default="single",
        help="Operation mode: 'single' to generate and display one plan, 'dataset' to generate multiple plans. Defaulted to 50.",
    )
    parser.add_argument(
        "--num-plans",
        type=int,
        default=50,
        help="Number of plans to generate (used in 'dataset' mode). Defaulted to 50.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to save the generated dataset (JSON format, used in 'dataset' mode). "
        "If not provided in dataset mode, a default name like 'dataset_NBblocks_NPplans.json' will be used. "
        "Set to 'NONE' (case-insensitive) to explicitly prevent saving. Defaulted to None.",
    )
    parser.add_argument(
        "--show-features",
        action="store_true",
        help="Display the mapping of feature names to their vector indices at the start. Defaulted to False.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. If None, results will vary. Defaulted to None.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging verbosity. Defaulted to INFO.",
    )

    args = parser.parse_args()

    setup_logging(args.log_level)  # Initialize logging first
    logger.info(f"Script execution started with arguments:\n{pformat(vars(args))}")

    if args.num_blocks <= 0:
        logger.error("--num-blocks must be a positive integer.")
        sys.exit(1)
    if args.mode == "dataset" and args.num_plans <= 0:
        logger.error("For 'dataset' mode, --num-plans must be a positive integer.")
        sys.exit(1)

    if args.seed is not None:
        logger.info(f"Setting random seed to: {args.seed}")
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    try:
        logger.debug(f"Initializing BlocksWorldGenerator for {args.num_blocks} blocks...")
        generator = BlocksWorldGenerator(num_blocks=args.num_blocks)
        logger.info(f"BlocksWorldGenerator initialized successfully with {args.num_blocks} blocks.")
        logger.debug(f"Number of features for encoding: {len(generator.feature_names)}")

        if args.show_features:
            logger.info("** Feature Names (index -> name) **")
            # Log as a multi-line string within a single log entry for better readability
            feature_list_str = "\n" + "\n".join(
                [f"  {i}: {feature}" for i, feature in enumerate(generator.feature_names)]
            )
            logger.info(feature_list_str)
            logger.info("** End of Feature Names **")

        if args.mode == "single":
            logger.info("Operating in 'single' plan generation mode.")
            sample_plan = generator.generate_random_plan()
            if sample_plan:
                logger.success("Single random plan generated successfully.")
                display_single_plan_details(generator, sample_plan)
            else:
                logger.error(
                    f"Failed to generate a single plan for {args.num_blocks} blocks. "
                    "This can occur if the random start/goal states are hard to connect "
                    "or if the search space is very large. Try again or with fewer blocks."
                )

        elif args.mode == "dataset":
            logger.info(f"Operating in 'dataset' generation mode for {args.num_plans} plans.")

            actual_output_file_str = args.output_file
            if actual_output_file_str is None:  # Not specified by user
                actual_output_file_str = (
                    f"dataset_{args.num_blocks}blocks_{args.num_plans}plans.json"
                )
                logger.warning(
                    f"No --output-file specified for dataset. Defaulting to: '{actual_output_file_str}'. "
                    "To prevent saving, use --output-file NONE."
                )

            if actual_output_file_str and actual_output_file_str.upper() == "NONE":
                actual_output_file_str = None
                logger.info("Dataset will be generated but not saved as per '--output-file NONE'.")

            generate_and_save_dataset(generator, args.num_plans, actual_output_file_str)

    except Exception:  # Catch all other exceptions
        logger.exception(
            "An critical error occurred during script execution:"
        )  # Automatically includes traceback
        sys.exit(1)

    logger.success("Script finished successfully.")


if __name__ == "__main__":
    main()
