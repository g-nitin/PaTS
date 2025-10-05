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

# GET_PROBLEM_HASH_SCRIPT: Python script to get unique hash for a problem
GET_PROBLEM_HASH_SCRIPT="./data/get_problem_hash.py"

# DOMAIN_NAME: Name of the planning domain (e.g., blocksworld)
DOMAIN_NAME="blocksworld"

# ENCODING_TYPE: The state encoding to use. Options: "bin", "sas"
ENCODING_TYPE="sas"

# MIN_BLOCKS & MAX_BLOCKS: Range of block numbers for problem generation
MIN_BLOCKS_TO_GENERATE=3
MAX_BLOCKS_TO_GENERATE=3

# This will be the root for raw_problems and processed_trajectories
BASE_DATA_DIR="./data"

# PROBLEMS_PER_CONFIG: Number of problems to generate for each block count
PROBLEMS_PER_CONFIG=1000

# FD_TIMEOUT: Timeout for Fast Downward (e.g., 60s, 5m)
FD_TIMEOUT="60s"

# FD_SEARCH_CONFIG: Fast Downward search configuration
# Common ones: "astar(lmcut())", "astar(ipdb())", "astar(blind())"
FD_SEARCH_CONFIG="astar(lmcut())"

# Helper Script Check
if [ ! -f "$PARSER_ENCODER_SCRIPT" ] || [ ! -f "$ANALYZE_AND_SPLIT_SCRIPT" ] || [ ! -f "$GET_PROBLEM_HASH_SCRIPT" ]; then
    echo "Error: Required Python scripts not found. Ensure they exist and are executable."
    exit 1
fi

# Define root directories for raw and processed data
RAW_PROBLEMS_ROOT="${BASE_DATA_DIR}/raw_problems/${DOMAIN_NAME}"
PROCESSED_TRAJECTORIES_ROOT="${BASE_DATA_DIR}/processed_trajectories/${DOMAIN_NAME}"

mkdir -p "$RAW_PROBLEMS_ROOT"
mkdir -p "$PROCESSED_TRAJECTORIES_ROOT"

TOTAL_SUCCESSFUL=0
TOTAL_FAILED_GENERATION=0
TOTAL_FAILED_FD=0
TOTAL_FAILED_VAL=0
TOTAL_FAILED_ENCODING=0
TOTAL_DUPLICATES_FILTERED=0

echo "Starting dataset generation..."
echo "Domain PDDL: $DOMAIN_FILE"
echo "Raw Problems Root: $RAW_PROBLEMS_ROOT"
echo "Processed Trajectories Root: $PROCESSED_TRAJECTORIES_ROOT"
echo "Encoding Type: $ENCODING_TYPE"
echo "***********************************"

for num_blocks in $(seq $MIN_BLOCKS_TO_GENERATE $MAX_BLOCKS_TO_GENERATE); do
    # Define block-specific directories for raw and processed data
    RAW_BLOCK_DIR="${RAW_PROBLEMS_ROOT}/N${num_blocks}"
    PROCESSED_BLOCK_ENCODING_DIR="${PROCESSED_TRAJECTORIES_ROOT}/N${num_blocks}/${ENCODING_TYPE}"

    # Ask user about overwriting if the processed outputs already exists
    if [ -d "$PROCESSED_BLOCK_ENCODING_DIR" ]; then
        read -p "Processed trajectories directory '$PROCESSED_BLOCK_ENCODING_DIR' already exists. Overwrite contents of subfolders for selected block sizes? (y/n): " overwrite_choice
        if [ "$overwrite_choice" != "y" ]; then
            echo "Skipping generation for $num_blocks blocks."
            continue
        fi
    fi
    
    echo "Generating problems for $num_blocks blocks into $RAW_BLOCK_DIR and $PROCESSED_BLOCK_ENCODING_DIR..."

    mkdir -p "$RAW_BLOCK_DIR/pddl"
    mkdir -p "$RAW_BLOCK_DIR/plans"
    mkdir -p "$RAW_BLOCK_DIR/val_out"
    mkdir -p "$RAW_BLOCK_DIR/trajectories_text"
    mkdir -p "$RAW_BLOCK_DIR/splits" # For train/val/test files
    mkdir -p "$PROCESSED_BLOCK_ENCODING_DIR"

    successful_for_size=0

    # Temporary file to store unique problem hashes for the current num_blocks
    UNIQUE_HASHES_FILE="${RAW_BLOCK_DIR}/.unique_problem_hashes_N${num_blocks}.tmp"

    # Define the ordered list of all possible predicates for this num_blocks
    # This is crucial for consistent binary encoding.
    # The parse_and_encode.py script will need to generate this list based on num_blocks.
    # Or, you can pre-generate these lists and pass the file path to the script.

    for i in $(seq 1 $PROBLEMS_PER_CONFIG); do
        SEED=$(( (num_blocks * 1000) + i )) # Simple way to get different seeds
        PROBLEM_BASENAME="blocks_${num_blocks}_problem_${i}"

        PDDL_FILE="${RAW_BLOCK_DIR}/pddl/${PROBLEM_BASENAME}.pddl"
        PLAN_FILE="${RAW_BLOCK_DIR}/plans/${PROBLEM_BASENAME}.plan"
        VAL_OUTPUT_FILE="${RAW_BLOCK_DIR}/val_out/${PROBLEM_BASENAME}.val.log"

        # For parse_and_encode.py outputs
        TEXT_TRAJECTORY_FILE="${RAW_BLOCK_DIR}/trajectories_text/${PROBLEM_BASENAME}.traj.txt"
        BINARY_TRAJECTORY_FILE_PREFIX="${PROCESSED_BLOCK_ENCODING_DIR}/${PROBLEM_BASENAME}" # .traj.<encoding>.npy will be appended

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
            --binary_output_prefix "$BINARY_TRAJECTORY_FILE_PREFIX" \
            --raw_data_dir "$RAW_BLOCK_DIR" # Manifest/info goes into raw_problems/blocksworld/N<num_blocks> dir

        if [ $? -ne 0 ]; then
            echo "    ERROR: Parsing/Encoding failed for $PROBLEM_BASENAME"
            TOTAL_FAILED_ENCODING=$((TOTAL_FAILED_ENCODING + 1))
            continue
        fi

        # Check for uniqueness based on initial/goal states from the generated text trajectory
        PROBLEM_HASH=$(uv run python "$GET_PROBLEM_HASH_SCRIPT" "$TEXT_TRAJECTORY_FILE")
        if [ $? -ne 0 ] || [ -z "$PROBLEM_HASH" ]; then
            echo "    ERROR: Failed to get problem hash for $PROBLEM_BASENAME. Skipping."
            TOTAL_FAILED_ENCODING=$((TOTAL_FAILED_ENCODING + 1)) # Count as encoding failure
            # Clean up files generated so far for this failed problem
            rm -f "$PDDL_FILE" "$PLAN_FILE" "$VAL_OUTPUT_FILE" "$TEXT_TRAJECTORY_FILE" \
                  "${BINARY_TRAJECTORY_FILE_PREFIX}.traj.${ENCODING_TYPE}.npy" \
                  "${BINARY_TRAJECTORY_FILE_PREFIX}.goal.${ENCODING_TYPE}.npy"
            continue
        fi

        if grep -q "$PROBLEM_HASH" "$UNIQUE_HASHES_FILE"; then
            echo "    DUPLICATE: Problem $PROBLEM_BASENAME (hash $PROBLEM_HASH) is a duplicate. Removing files."
            TOTAL_DUPLICATES_FILTERED=$((TOTAL_DUPLICATES_FILTERED + 1))
            # Clean up all files generated for this duplicate problem
            rm -f "$PDDL_FILE" "$PLAN_FILE" "$VAL_OUTPUT_FILE" "$TEXT_TRAJECTORY_FILE" \
                  "${BINARY_TRAJECTORY_FILE_PREFIX}.traj.${ENCODING_TYPE}.npy" \
                  "${BINARY_TRAJECTORY_FILE_PREFIX}.goal.${ENCODING_TYPE}.npy"
            # Do NOT increment successful_for_size or TOTAL_SUCCESSFUL
        else
            # If unique, add hash to tracker and count as successful
            echo "$PROBLEM_HASH" >> "$UNIQUE_HASHES_FILE"
            echo "    SUCCESS: $PROBLEM_BASENAME processed and is unique."
            successful_for_size=$((successful_for_size + 1))
            TOTAL_SUCCESSFUL=$((TOTAL_SUCCESSFUL + 1))
        fi
    
    done
    echo -e "\n"
    echo "  Finished $num_blocks blocks. Successful: $successful_for_size / $PROBLEMS_PER_CONFIG"
    rm -f "$UNIQUE_HASHES_FILE" # Clean up temporary hash file

    # 5. Analyze dataset and create train-test splits
    # Call to `analyze_dataset_splits.py` with the argument `raw_block_dir`
    echo -e "\n"
    echo "Analyzing dataset splits for $num_blocks blocks..."
    uv run python "$ANALYZE_AND_SPLIT_SCRIPT" \
        "$RAW_BLOCK_DIR"

    if [ $? -ne 0 ]; then
        echo "ERROR: Dataset analysis failed for $PROBLEM_BASENAME"
        TOTAL_FAILED_ENCODING=$((TOTAL_FAILED_ENCODING + 1))
        # Do not continue, as this is a critical step for the entire N-block dataset
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
echo "  Total Duplicate Problems Filtered: $TOTAL_DUPLICATES_FILTERED"
echo "***********************************"