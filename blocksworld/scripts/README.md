- `generate_plans.py`:

    This script serves as an interactor for `BlocksWorldGenerator`. To use this script:

    -   Generate and display a single plan for 3 blocks (default log level INFO):
        ```bash
        python generate_plans.py --num-blocks 3
        ```
    -   Generate a single plan, show features, and use DEBUG log level:
        ```bash
        python generate_plans.py --num-blocks 3 --show-features --log-level DEBUG
        ```
    -   Generate a dataset of 100 plans for 4 blocks and save it:
        ```bash
        python generate_plans.py --num-blocks 4 --mode dataset --num-plans 100 --output-file my_dataset_4blocks.json
        ```
    -   Generate a dataset of 10 plans for 2 blocks, use a specific seed, and don't save the file:
        ```bash
        python generate_plans.py --num-blocks 2 --mode dataset --num-plans 10 --seed 42 --output-file NONE
        ```
    -   Get help on command-line options:
        ```bash
        python generate_plans.py --help
        ```
