#!/bin/bash

# ROOT_DIR: Base directory for all planning related tools and files
ROOT_DIR="$HOME/usc/ai4s/libraries/planning/"

# DOMAIN_FILE: Path to the PDDL domain file (e.g., blocksworld domain)
DOMAIN_FILE="${ROOT_DIR}pddl-generators/blocksworld/4ops/domain.pddl"

# PROBLEM_GENERATOR_SCRIPT: Path to the script that generates PDDL problem files
# This is the executable from the pddl-generators repository for blocksworld
PROBLEM_GENERATOR_SCRIPT="${ROOT_DIR}pddl-generators/blocksworld/blocksworld"

# FD_PATH: Path to the Fast Downward planner script
FD_PATH="${ROOT_DIR}downward/fast-downward.py"

# VALIDATE_PATH: Path to the VAL executable
VALIDATE_PATH="${ROOT_DIR}VAL/validate"

# PARSER_ENCODER_SCRIPT: Python script to parse VAL output, PDDL, and encode states
PARSER_ENCODER_SCRIPT="./data/parse_and_encode.py"

# ANALYZE_AND_SPLIT_SCRIPT: Python script to analyze dataset and create train-test splits
ANALYZE_AND_SPLIT_SCRIPT="./data/analyze_dataset_splits.py"

# OUTPUT_DIR: Directory to store generated PDDL problems, plans, VAL outputs, and final trajectories
OUTPUT_DIR="./data-sas"

# MIN_BLOCKS & MAX_BLOCKS: Range of block numbers for problem generation
MIN_BLOCKS_TO_GENERATE=4
MAX_BLOCKS_TO_GENERATE=4

# PROBLEMS_PER_CONFIG: Number of problems to generate for each block count
PROBLEMS_PER_CONFIG=10000

# FD_TIMEOUT: Timeout for Fast Downward (e.g., 60s, 5m)
FD_TIMEOUT="60s"

# FD_SEARCH_CONFIG: Fast Downward search configuration
# Common ones: "astar(lmcut())", "astar(ipdb())", "astar(blind())"
FD_SEARCH_CONFIG="astar(lmcut())"

# ENCODING_TYPE: The state encoding to use. Options: "bin", "sas"
ENCODING_TYPE="sas"

# Helper Script Check
if [ ! -f "$PARSER_ENCODER_SCRIPT" ]; then
    echo "Error: Parser/Encoder script '$PARSER_ENCODER_SCRIPT' not found."
    echo "Please ensure it exists and is executable."
    exit 1
fi
# Ensure output directory exists
# If it exists, ask the user if they want to overwrite it or use a new one
if [ -d "$OUTPUT_DIR" ]; then
    read -p "Output directory '$OUTPUT_DIR' already exists. Overwrite contents of subfolders for selected block sizes? (y/n): " overwrite_choice
    if [ "$overwrite_choice" != "y" ]; then
        read -p "Enter new base output directory name (e.g., data_new): " new_output_dir
        if [ -n "$new_output_dir" ]; then # Check if user provided a new name
        OUTPUT_DIR="./$new_output_dir"
        else
            echo "No new directory name provided. Exiting."
            exit 1
        fi
    fi
fi

mkdir -p "$OUTPUT_DIR"
# Subdirectories will be created inside the block-specific folders

TOTAL_SUCCESSFUL=0
TOTAL_FAILED_GENERATION=0
TOTAL_FAILED_FD=0
TOTAL_FAILED_VAL=0
TOTAL_FAILED_ENCODING=0

echo "Starting dataset generation..."
echo "Domain: $DOMAIN_FILE"
echo "Base Output Directory: $OUTPUT_DIR"
echo "***********************************"

for num_blocks in $(seq $MIN_BLOCKS_TO_GENERATE $MAX_BLOCKS_TO_GENERATE); do
    # BLOCK_SPECIFIC_DATA_DIR is where manifest, pddl, plans, trajectories for this num_blocks go
    BLOCK_SPECIFIC_DATA_DIR="${OUTPUT_DIR}/blocks_${num_blocks}".
    echo "Generating problems for $num_blocks blocks into $BLOCK_SPECIFIC_DATA_DIR..."

    mkdir -p "$BLOCK_SPECIFIC_DATA_DIR/pddl"
    mkdir -p "$BLOCK_SPECIFIC_DATA_DIR/plans"
    mkdir -p "$BLOCK_SPECIFIC_DATA_DIR/val_out"
    mkdir -p "$BLOCK_SPECIFIC_DATA_DIR/trajectories_text"
    mkdir -p "$BLOCK_SPECIFIC_DATA_DIR/trajectories_bin"

    successful_for_size=0

    # Define the ordered list of all possible predicates for this num_blocks
    # This is crucial for consistent binary encoding.
    # The parse_and_encode.py script will need to generate this list based on num_blocks.
    # Or, you can pre-generate these lists and pass the file path to the script.

    for i in $(seq 1 $PROBLEMS_PER_CONFIG); do
        SEED=$(( (num_blocks * 1000) + i )) # Simple way to get different seeds
        PROBLEM_BASENAME="blocks_${num_blocks}_problem_${i}"

        PDDL_FILE="${BLOCK_SPECIFIC_DATA_DIR}/pddl/${PROBLEM_BASENAME}.pddl"
        PLAN_FILE="${BLOCK_SPECIFIC_DATA_DIR}/plans/${PROBLEM_BASENAME}.plan"
        VAL_OUTPUT_FILE="${BLOCK_SPECIFIC_DATA_DIR}/val_out/${PROBLEM_BASENAME}.val.log"

        # For parse_and_encode.py outputs
        TEXT_TRAJECTORY_FILE="${BLOCK_SPECIFIC_DATA_DIR}/trajectories_text/${PROBLEM_BASENAME}.traj.txt"
        BINARY_TRAJECTORY_FILE="${BLOCK_SPECIFIC_DATA_DIR}/trajectories_bin/${PROBLEM_BASENAME}.traj.bin" # Or .npz, .pt, etc.
        GOAL_BINARY_FILE="${BLOCK_SPECIFIC_DATA_DIR}/trajectories_bin/${PROBLEM_BASENAME}.goal.bin"

        echo -e "\n"
        echo "  Processing: $PROBLEM_BASENAME (Seed: $SEED)"

        # 1. Generate PDDL problem
        # The blocksworld generator takes: <ops_mode (3 or 4)> <num_blocks> <seed>
        echo "    Generating PDDL for $PROBLEM_BASENAME with $num_blocks blocks..."
        "$PROBLEM_GENERATOR_SCRIPT" 4 "$num_blocks" "$SEED" > "$PDDL_FILE"
        if [ $? -ne 0 ] || [ ! -s "$PDDL_FILE" ]; then
            echo "    ERROR: Failed to generate PDDL for $PROBLEM_BASENAME"
            TOTAL_FAILED_GENERATION=$((TOTAL_FAILED_GENERATION + 1))
            rm -f "$PDDL_FILE" # Clean up empty/failed file
            continue
        fi

        # 2. Solve with Fast Downward (with timeout)
        echo "    Running Fast Downward for $PROBLEM_BASENAME..."
        timeout "$FD_TIMEOUT" "$FD_PATH" --plan-file "$PLAN_FILE" "$DOMAIN_FILE" "$PDDL_FILE" --search "$FD_SEARCH_CONFIG" > /dev/null 2>&1
        
        if [ ! -s "$PLAN_FILE" ]; then
            echo "    WARNING: FD failed or timed out for $PROBLEM_BASENAME"
            echo "    Command was : $FD_PATH --plan-file $PLAN_FILE $DOMAIN_FILE $PDDL_FILE --search $FD_SEARCH_CONFIG"
            TOTAL_FAILED_FD=$((TOTAL_FAILED_FD + 1))
            rm -f "$PLAN_FILE" # Clean up empty plan file
            rm -f "$PDDL_FILE" # Clean problem file
            continue
        fi

        # Also skip on plan with no length (1 line containing "; cost = 0 (unit cost)")
        PLAN_LENGTH=$(wc -l < "$PLAN_FILE")
        if [ "$PLAN_LENGTH" -eq 1 ]; then
            echo "    WARNING: FD produced an empty plan for $PROBLEM_BASENAME"
            TOTAL_FAILED_FD=$((TOTAL_FAILED_FD + 1))
            rm -f "$PLAN_FILE" # Clean up empty plan file
            rm -f "$PDDL_FILE" # Clean problem file
            continue
        fi

        # 3. Validate plan with VAL (verbose mode)
        echo "    Validating plan with VAL for $PROBLEM_BASENAME..."
        "$VALIDATE_PATH" -v "$DOMAIN_FILE" "$PDDL_FILE" "$PLAN_FILE" > "$VAL_OUTPUT_FILE" 2>&1
        # Check VAL's exit code and output for success
        # VAL usually exits 0 on success. "Plan valid" or "Plan executed successfully" should be in output.
        if [ $? -ne 0 ] || ! grep -q -E "Plan valid|Plan executed successfully" "$VAL_OUTPUT_FILE"; then
            echo "    WARNING: VAL validation failed or plan invalid for $PROBLEM_BASENAME"
            cat "$VAL_OUTPUT_FILE" # Optional: print VAL output for debugging
            TOTAL_FAILED_VAL=$((TOTAL_FAILED_VAL + 1))
            continue
        fi

        # 4. Parse VAL output, PDDL, encode states, and save trajectory
        echo "    Encoding trajectory for $PROBLEM_BASENAME using '$ENCODING_TYPE' encoding..."
        uv run python "$PARSER_ENCODER_SCRIPT" \
            --val_output_file "$VAL_OUTPUT_FILE" \
            --pddl_domain_file "$DOMAIN_FILE" \
            --pddl_problem_file "$PDDL_FILE" \
            --num_blocks "$num_blocks" \
            --encoding_type "$ENCODING_TYPE" \
            --text_trajectory_output "$TEXT_TRAJECTORY_FILE" \
            --binary_trajectory_output "$BINARY_TRAJECTORY_FILE" \
            --binary_goal_output "$GOAL_BINARY_FILE" \
            --manifest_output_dir "$BLOCK_SPECIFIC_DATA_DIR" # Manifest/info goes into blocks_N dir
        
        if [ $? -ne 0 ]; then
            echo "    ERROR: Parsing/Encoding failed for $PROBLEM_BASENAME"
            TOTAL_FAILED_ENCODING=$((TOTAL_FAILED_ENCODING + 1))
            continue
        fi

        # If we reach here, everything was successful for this problem
        echo "    SUCCESS: $PROBLEM_BASENAME processed."
        successful_for_size=$((successful_for_size + 1))
        TOTAL_SUCCESSFUL=$((TOTAL_SUCCESSFUL + 1))
    done
    echo -e "\n"
    echo "  Finished $num_blocks blocks. Successful: $successful_for_size / $PROBLEMS_PER_CONFIG"

    # 5. Analyze dataset and create train-test splits
    # Call to `analyze_dataset_splits.py` with arguments: 
        # `dataset_dir`: Path to the root directory of the generated dataset (containing 'plans' subdirectory).
        # `output_dir`: Directory to save the split files (train_files.txt, etc.) and distribution plots.
    echo -e "\n"
    echo "Analyzing dataset splits for $num_blocks blocks..."
    uv run python "$ANALYZE_AND_SPLIT_SCRIPT" \
        "$BLOCK_SPECIFIC_DATA_DIR" "$BLOCK_SPECIFIC_DATA_DIR"

    if [ $? -ne 0 ]; then
        echo "ERROR: Dataset analysis failed for $PROBLEM_BASENAME"
        TOTAL_FAILED_ENCODING=$((TOTAL_FAILED_ENCODING + 1))
        continue
    fi
    echo "***********************************"
    echo -e "\n"
done

echo "Dataset Generation Complete."
echo "Summary:"
echo "  Total Successfully Processed: $TOTAL_SUCCESSFUL"
echo "  Total Failed PDDL Generation: $TOTAL_FAILED_GENERATION"
echo "  Total Failed Fast Downward:   $TOTAL_FAILED_FD"
echo "  Total Failed VAL Validation:  $TOTAL_FAILED_VAL"
echo "  Total Failed Encoding:        $TOTAL_FAILED_ENCODING"
echo "***********************************"