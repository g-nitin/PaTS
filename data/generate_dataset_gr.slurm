#!/bin/bash
#SBATCH --job-name=pddl-dset
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

# %x gives job name
# %A Job array's master job allocation number.
# %a Job array ID (index) number.
# %j Job allocation number.
# %N Node name. Only one file is created, so %N will be replaced by the name of the first node in the job, which is the one that runs the script.
# %u User name.

#SBATCH --mail-user=niting@email.sc.edu
#SBATCH --mail-type=ALL

#SBATCH --partition=wholenode
#SBATCH --time=96:00:00
# Max time for `wholenode` partition is 96 hours (4 days)

# $HOME: /home/x-ngupta6
module use "$HOME/privatemodules"
module load conda-env/pats-py3.12.8
conda activate pats

TARGET_UNIQUE_PROBLEMS_PER_CONFIG=$((10 * 10)) # Aim for this many unique problems per block/gripper configuration
MAX_GENERATION_ATTEMPTS_PER_CONFIG=2000        # Stop after this many attempts, even if target not met
NUM_ROBOTS=2
NUM_ROOMS=4
NUM_OBJECTS=3

DOMAIN_NAME="grippers"
ENCODING_TYPES=("bin" "sas")
ROOT_DIR="$HOME/planning/"                                              # Base directory for all planning related tools and files
DOMAIN_FILE="${ROOT_DIR}pddl-generators/grippers/domain.pddl"           # Path to the PDDL domain file
FD_PATH="${ROOT_DIR}downward/fast-downward.py"                          # Path to the Fast Downward planner script
VALIDATE_PATH="${ROOT_DIR}VAL/bin/Validate"                             # Path to the VAL executable
PARSER_ENCODER_SCRIPT="./data/parse_and_encode_gr.py"                   # Python script to parse VAL output, PDDL, and encode states
ANALYZE_AND_SPLIT_SCRIPT="./data/analyze_dataset_splits.py"             # Python script to analyze dataset and create train-test splits
GET_PROBLEM_HASH_SCRIPT="./data/get_problem_hash.py"                    # Python script to get unique hash for a problem
PROBLEM_GENERATOR_SCRIPT="${ROOT_DIR}pddl-generators/grippers/grippers" # Path to the script that generates PDDL problem files (executable from the pddl-generators repository)

FD_TIMEOUT="60s"                  # Timeout for Fast Downward (e.g., 60s, 5m)
FD_SEARCH_CONFIG="astar(lmcut())" # FS search configuration; common ones: "astar(lmcut())", "astar(ipdb())", "astar(blind())"

# Helper Script Check
if [ ! -f "$PARSER_ENCODER_SCRIPT" ] || [ ! -f "$ANALYZE_AND_SPLIT_SCRIPT" ] || [ ! -f "$GET_PROBLEM_HASH_SCRIPT" ]; then
    echo "Error: Required Python scripts not found. Ensure they exist and are executable."
    exit 1
fi

BASE_DATA_DIR="./data" # This will be the root for raw_problems and processed_trajectories
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

echo "Starting dataset generation for $DOMAIN_NAME..."
echo "Domain PDDL: $DOMAIN_FILE"
echo "Raw Problems Root: $RAW_PROBLEMS_ROOT"
echo "Processed Trajectories Root: $PROCESSED_TRAJECTORIES_ROOT"
echo "Encoding Types: ${ENCODING_TYPES[*]}"
echo "***********************************"
echo ""

PROBLEM_CONFIG_NAME="R${NUM_ROBOTS}-O${NUM_OBJECTS}-RM${NUM_ROOMS}"
RAW_PROBLEM_DIR="${RAW_PROBLEMS_ROOT}/${PROBLEM_CONFIG_NAME}"
PROCESSED_PROBLEM_DIR="${PROCESSED_TRAJECTORIES_ROOT}/${PROBLEM_CONFIG_NAME}"

# Ask user about overwriting if the processed outputs already exists
if [ -d "$PROCESSED_PROBLEM_DIR" ]; then
    echo "Processed trajectories directory '$PROCESSED_PROBLEM_DIR' already exists."
    echo "Skipping generation for this configuration."
    exit 0 # Use exit instead of continue since there's no loop
fi

echo "Generating problems for config ${PROBLEM_CONFIG_NAME} into $RAW_PROBLEM_DIR..."

mkdir -p "$RAW_PROBLEM_DIR/pddl"
mkdir -p "$RAW_PROBLEM_DIR/plans"
mkdir -p "$RAW_PROBLEM_DIR/val_out"
mkdir -p "$RAW_PROBLEM_DIR/trajectories_text"
mkdir -p "$RAW_PROBLEM_DIR/splits" # For train/val/test files

# Initialize counters for the loop
current_attempt=0
successful_for_size=0 # This counter will track UNIQUE successful problems

# Temporary file to store unique problem hashes for the current num_blocks/config
# Ensure this file is created/truncated at the start of each config loop
UNIQUE_HASHES_FILE="${RAW_PROBLEM_DIR}/.unique_problem_hashes.tmp"
>"$UNIQUE_HASHES_FILE" # Create or truncate the file

while [ "$successful_for_size" -lt "$TARGET_UNIQUE_PROBLEMS_PER_CONFIG" ] && [ "$current_attempt" -lt "$MAX_GENERATION_ATTEMPTS_PER_CONFIG" ]; do
    current_attempt=$((current_attempt + 1))
    SEED=$(((NUM_ROBOTS * 10000) + (NUM_ROOMS * 1000) + (NUM_OBJECTS * 100) + current_attempt))

    PROBLEM_BASENAME="grippers_${PROBLEM_CONFIG_NAME}_problem_${current_attempt}"
    PDDL_FILE="${RAW_PROBLEM_DIR}/pddl/${PROBLEM_BASENAME}.pddl"
    PLAN_FILE="${RAW_PROBLEM_DIR}/plans/${PROBLEM_BASENAME}.plan"
    VAL_OUTPUT_FILE="${RAW_PROBLEM_DIR}/val_out/${PROBLEM_BASENAME}.val.log"

    # For parse_and_encode.py outputs
    TEXT_TRAJECTORY_FILE="${RAW_PROBLEM_DIR}/trajectories_text/${PROBLEM_BASENAME}.traj.txt"
    BINARY_TRAJECTORY_FILE_PREFIX="${PROCESSED_BLOCK_ENCODING_DIR}/${PROBLEM_BASENAME}" # .traj.<encoding>.npy will be appended

    echo ""
    echo "  Processing: $PROBLEM_BASENAME (Seed: $SEED)"

    # 1. Generate PDDL problem
    echo "    Generating PDDL for $PROBLEM_BASENAME"
    "$PROBLEM_GENERATOR_SCRIPT" -n "$NUM_ROBOTS" -r "$NUM_ROOMS" -o "$NUM_OBJECTS" -s "$SEED" >"$PDDL_FILE"
    if [ $? -ne 0 ] || [ ! -s "$PDDL_FILE" ]; then
        echo "    ERROR: Failed to generate PDDL for $PROBLEM_BASENAME"
        TOTAL_FAILED_GENERATION=$((TOTAL_FAILED_GENERATION + 1))
        rm -f "$PDDL_FILE" # Clean up empty/failed file
        continue
    fi

    # 2. Solve with Fast Downward (with timeout)
    echo "    Running Fast Downward for $PROBLEM_BASENAME..."
    timeout "$FD_TIMEOUT" "$FD_PATH" --plan-file "$PLAN_FILE" "$DOMAIN_FILE" "$PDDL_FILE" --search "$FD_SEARCH_CONFIG" >/dev/null 2>&1

    if [ ! -s "$PLAN_FILE" ]; then
        echo "    WARNING: FD failed or timed out for $PROBLEM_BASENAME"
        echo "    Command was : $FD_PATH --plan-file $PLAN_FILE $DOMAIN_FILE $PDDL_FILE --search $FD_SEARCH_CONFIG"
        TOTAL_FAILED_FD=$((TOTAL_FAILED_FD + 1))
        rm -f "$PLAN_FILE" # Clean up empty plan file
        rm -f "$PDDL_FILE" # Clean problem file
        continue
    fi

    # Also skip on plan with no length (1 line containing "; cost = 0 (unit cost)")
    PLAN_LENGTH=$(wc -l <"$PLAN_FILE")
    if [ "$PLAN_LENGTH" -eq 1 ]; then
        echo "    WARNING: FD produced an empty plan for $PROBLEM_BASENAME"
        TOTAL_FAILED_FD=$((TOTAL_FAILED_FD + 1))
        rm -f "$PLAN_FILE" # Clean up empty plan file
        rm -f "$PDDL_FILE" # Clean problem file
        continue
    fi

    # 3. Validate plan with VAL (verbose mode)
    echo "    Validating plan with VAL for $PROBLEM_BASENAME..."
    "$VALIDATE_PATH" -v "$DOMAIN_FILE" "$PDDL_FILE" "$PLAN_FILE" >"$VAL_OUTPUT_FILE" 2>&1
    # Check VAL's exit code and output for success
    # VAL usually exits 0 on success. "Plan valid" or "Plan executed successfully" should be in output.
    if [ $? -ne 0 ] || ! grep -q -E "Plan valid|Plan executed successfully" "$VAL_OUTPUT_FILE"; then
        echo "    WARNING: VAL validation failed or plan invalid for $PROBLEM_BASENAME"
        cat "$VAL_OUTPUT_FILE" # Optional: print VAL output for debugging
        TOTAL_FAILED_VAL=$((TOTAL_FAILED_VAL + 1))
        continue
    fi

    # 4. Parse and Encode for ALL specified types
    TEXT_TRAJECTORY_FILE="${RAW_PROBLEM_DIR}/trajectories_text/${PROBLEM_BASENAME}.traj.txt"
    PARSE_SUCCESSFUL=false

    for enc_type in "${ENCODING_TYPES[@]}"; do
        PROCESSED_BLOCK_ENCODING_DIR="${PROCESSED_PROBLEM_DIR}/${enc_type}"
        mkdir -p "$PROCESSED_BLOCK_ENCODING_DIR"
        BINARY_TRAJECTORY_FILE_PREFIX="${PROCESSED_BLOCK_ENCODING_DIR}/${PROBLEM_BASENAME}"

        echo "    Encoding trajectory for $PROBLEM_BASENAME using '$enc_type' encoding..."
        python "$PARSER_ENCODER_SCRIPT" \
            --val_output_file "$VAL_OUTPUT_FILE" \
            --pddl_domain_file "$DOMAIN_FILE" \
            --pddl_problem_file "$PDDL_FILE" \
            --num-robots "$NUM_ROBOTS" \
            --num-rooms "$NUM_ROOMS" \
            --num-objects "$NUM_OBJECTS" \
            --encoding_type "$enc_type" \
            --text_trajectory_output "$TEXT_TRAJECTORY_FILE" \
            --binary_output_prefix "$BINARY_TRAJECTORY_FILE_PREFIX"

        if [ $? -eq 0 ]; then
            PARSE_SUCCESSFUL=true
        else
            echo "    ERROR: Parsing/Encoding for '$enc_type' failed for $PROBLEM_BASENAME"
        fi
    done

    if ! $PARSE_SUCCESSFUL; then
        echo "    ERROR: All parsing/encoding attempts failed for $PROBLEM_BASENAME. Cleaning up raw files."
        TOTAL_FAILED_ENCODING=$((TOTAL_FAILED_ENCODING + 1))
        rm -f "$PDDL_FILE" "$PLAN_FILE" "$VAL_OUTPUT_FILE"
        continue
    fi

    # 5. Check for uniqueness based on the generated text trajectory
    PROBLEM_HASH=$(python "$GET_PROBLEM_HASH_SCRIPT" "$TEXT_TRAJECTORY_FILE")
    if [ $? -ne 0 ] || [ -z "$PROBLEM_HASH" ]; then
        echo "    ERROR: Failed to get problem hash for $PROBLEM_BASENAME. Skipping."
        TOTAL_FAILED_ENCODING=$((TOTAL_FAILED_ENCODING + 1))
        # Clean up files generated so far for this failed problem
        rm -f "$PDDL_FILE" "$PLAN_FILE" "$VAL_OUTPUT_FILE" "$TEXT_TRAJECTORY_FILE"
        for enc_type in "${ENCODING_TYPES[@]}"; do
            PROCESSED_BLOCK_ENCODING_DIR="${PROCESSED_PROBLEM_DIR}/${enc_type}"
            rm -f "${PROCESSED_BLOCK_ENCODING_DIR}/${PROBLEM_BASENAME}.traj.${enc_type}.npy" \
                "${PROCESSED_BLOCK_ENCODING_DIR}/${PROBLEM_BASENAME}.goal.${enc_type}.npy"
        done
        continue
    fi

    if grep -q "$PROBLEM_HASH" "$UNIQUE_HASHES_FILE"; then
        echo "    DUPLICATE: Problem $PROBLEM_BASENAME (hash $PROBLEM_HASH) is a duplicate. Removing files."
        TOTAL_DUPLICATES_FILTERED=$((TOTAL_DUPLICATES_FILTERED + 1))
        # Clean up all files generated for this duplicate problem
        rm -f "$PDDL_FILE" "$PLAN_FILE" "$VAL_OUTPUT_FILE" "$TEXT_TRAJECTORY_FILE"
        for enc_type in "${ENCODING_TYPES[@]}"; do
            PROCESSED_BLOCK_ENCODING_DIR="${PROCESSED_PROBLEM_DIR}/${enc_type}"
            rm -f "${PROCESSED_BLOCK_ENCODING_DIR}/${PROBLEM_BASENAME}.traj.${enc_type}.npy" \
                "${PROCESSED_BLOCK_ENCODING_DIR}/${PROBLEM_BASENAME}.goal.${enc_type}.npy"
        done
    else
        # If unique, add hash to tracker and count as successful
        echo "$PROBLEM_HASH" >>"$UNIQUE_HASHES_FILE"
        echo "    SUCCESS: $PROBLEM_BASENAME processed and is unique. ($successful_for_size / $TARGET_UNIQUE_PROBLEMS_PER_CONFIG unique problems so far)"
        successful_for_size=$((successful_for_size + 1))
        TOTAL_SUCCESSFUL=$((TOTAL_SUCCESSFUL + 1))
    fi

done
echo ""
echo "  Finished processing. Successful unique problems: $successful_for_size / $TARGET_UNIQUE_PROBLEMS_PER_CONFIG (Total attempts: $current_attempt)"
rm -f "$UNIQUE_HASHES_FILE" # Clean up temporary hash file

# 5. Analyze dataset and create train-test splits
echo ""
echo "Analyzing dataset splits..."
python "$ANALYZE_AND_SPLIT_SCRIPT" "$RAW_PROBLEM_DIR"

if [ $? -ne 0 ]; then
    echo "ERROR: Dataset analysis failed for $PROBLEM_BASENAME"
    TOTAL_FAILED_ENCODING=$((TOTAL_FAILED_ENCODING + 1))
    # Do not continue, as this is a critical step
fi
echo "***********************************"
echo ""

echo ""
echo "***********************************"
echo "Dataset Generation Complete."
echo "Summary:"
echo "  Total Successfully Processed      : $TOTAL_SUCCESSFUL"
echo "  Total Failed PDDL Generation      : $TOTAL_FAILED_GENERATION"
echo "  Total Failed Fast Downward        : $TOTAL_FAILED_FD"
echo "  Total Failed VAL Validation       : $TOTAL_FAILED_VAL"
echo "  Total Failed Encoding             : $TOTAL_FAILED_ENCODING"
echo "  Total Duplicate Problems Filtered : $TOTAL_DUPLICATES_FILTERED"
echo "***********************************"
