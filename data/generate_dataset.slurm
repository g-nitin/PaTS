#!/bin/bash
#SBATCH --job-name=pddl-dset
#SBATCH --output=logs/%x_%A_%a_%j.out
#SBATCH --error=logs/%x_%A_%a_%j.err

#SBATCH --mail-user=niting@email.sc.edu
#SBATCH --mail-type=ALL

#SBATCH --partition=wholenode
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
# SBATCH --time=24:00:00
# Choose how many shards (array tasks) YOU want to run concurrently.
# Edit this line to scale out (e.g., 1-40).
#SBATCH --array=1-20

## USER KNOBS (EDIT THESE)
# Total problems to generate per block size (matches original script intention)
PROBLEMS_PER_CONFIG=100

# Problems per shard (array task). Total shards × CHUNK_SIZE should cover PROBLEMS_PER_CONFIG.
CHUNK_SIZE=5

# Per-node parallel workers inside each shard (usually match --cpus-per-task)
PARALLEL_JOBS=${SLURM_CPUS_ON_NODE:-16}

# Overwrite processed outputs if they exist? 1=yes (non-interactive), 0=skip that N
OVERWRITE_EXISTING=1

# PDDL block counts to generate (inclusive range)
MIN_BLOCKS_TO_GENERATE=6
MAX_BLOCKS_TO_GENERATE=6

# Fast Downward config
FD_TIMEOUT="60s"
FD_SEARCH_CONFIG="astar(lmcut())"

# Global dedup across shards (0=per-shard only, 1=shared file + flock)
USE_GLOBAL_DEDUP=1

## Environment / Modules
# $HOME: /home/x-ngupta6
module use "$HOME/privatemodules"
module load conda-env/pats-py3.12.8
conda activate pats

# Fail fast on undefined vars; don't exit on first error (we handle per-task errors)
set -u
umask 077
mkdir -p logs

## Paths / Config
ROOT_DIR="$HOME/planning/"

DOMAIN_FILE="${ROOT_DIR}pddl-generators/blocksworld/4ops/domain.pddl"
PROBLEM_GENERATOR_SCRIPT="${ROOT_DIR}pddl-generators/blocksworld/blocksworld"
FD_PATH="${ROOT_DIR}downward/fast-downward.py"
VALIDATE_PATH="${ROOT_DIR}VAL/bin/Validate"

PARSER_ENCODER_SCRIPT="./data/parse_and_encode.py"
ANALYZE_AND_SPLIT_SCRIPT="./data/analyze_dataset_splits.py"
GET_PROBLEM_HASH_SCRIPT="./data/get_problem_hash.py"

DOMAIN_NAME="blocksworld"
ENCODING_TYPE="sas"

BASE_DATA_DIR="./data"
RAW_PROBLEMS_ROOT="${BASE_DATA_DIR}/raw_problems/${DOMAIN_NAME}"
PROCESSED_TRAJECTORIES_ROOT="${BASE_DATA_DIR}/processed_trajectories/${DOMAIN_NAME}"

mkdir -p "$RAW_PROBLEMS_ROOT" "$PROCESSED_TRAJECTORIES_ROOT"

# Checks
for f in "$PARSER_ENCODER_SCRIPT" "$ANALYZE_AND_SPLIT_SCRIPT" "$GET_PROBLEM_HASH_SCRIPT" \
         "$PROBLEM_GENERATOR_SCRIPT" "$FD_PATH" "$VALIDATE_PATH" "$DOMAIN_FILE"; do
  [[ -e "$f" ]] || { echo "ERROR: Missing required file: $f"; exit 1; }
done

## Shard math (based on array index & CHUNK_SIZE)
TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
TOTAL_SHARDS="${SLURM_ARRAY_TASK_MAX:-1}"

# Compute start/end i for this shard (1-indexed problems)
# Example: CHUNK_SIZE=500, TASK_ID=3 => i in [1001..1500]
I_START=$(( (TASK_ID - 1) * CHUNK_SIZE + 1 ))
I_END=$(( TASK_ID * CHUNK_SIZE ))
if (( I_START > PROBLEMS_PER_CONFIG )); then
  echo "Shard $TASK_ID has no work (start=$I_START > total=$PROBLEMS_PER_CONFIG)."
  exit 0
fi
if (( I_END > PROBLEMS_PER_CONFIG )); then
  I_END=$PROBLEMS_PER_CONFIG
fi

echo "Shard $TASK_ID/$TOTAL_SHARDS handles problems i=[${I_START}..${I_END}]"
echo "PARALLEL_JOBS=${PARALLEL_JOBS}, FD_TIMEOUT=${FD_TIMEOUT}, FD_SEARCH_CONFIG=${FD_SEARCH_CONFIG}"

## Counters (for this shard)
TOTAL_SUCCESSFUL=0
TOTAL_FAILED_GENERATION=0
TOTAL_FAILED_FD=0
TOTAL_FAILED_VAL=0
TOTAL_FAILED_ENCODING=0
TOTAL_DUPLICATES_FILTERED=0

## Per-problem worker
process_problem() {
  local num_blocks="$1"
  local i="$2"

  local RAW_BLOCK_DIR="${RAW_PROBLEMS_ROOT}/N${num_blocks}"
  local PROCESSED_BLOCK_ENCODING_DIR="${PROCESSED_TRAJECTORIES_ROOT}/N${num_blocks}/${ENCODING_TYPE}"

  local SEED=$(( (num_blocks * 1000) + i ))
  local PROBLEM_BASENAME="blocks_${num_blocks}_problem_${i}"

  local PDDL_FILE="${RAW_BLOCK_DIR}/pddl/${PROBLEM_BASENAME}.pddl"
  local PLAN_FILE="${RAW_BLOCK_DIR}/plans/${PROBLEM_BASENAME}.plan"
  local VAL_OUTPUT_FILE="${RAW_BLOCK_DIR}/val_out/${PROBLEM_BASENAME}.val.log"
  local TEXT_TRAJECTORY_FILE="${RAW_BLOCK_DIR}/trajectories_text/${PROBLEM_BASENAME}.traj.txt"
  local BINARY_TRAJECTORY_FILE_PREFIX="${PROCESSED_BLOCK_ENCODING_DIR}/${PROBLEM_BASENAME}"

  local SHARD_HASHES_FILE="${RAW_BLOCK_DIR}/.unique_problem_hashes_N${num_blocks}.shard_${TASK_ID}.tmp"
  local GLOBAL_HASHES_FILE="${RAW_BLOCK_DIR}/.unique_problem_hashes_N${num_blocks}.global"

  echo "Processing: $PROBLEM_BASENAME (Seed: $SEED)"

  # 1) Generate PDDL
  "$PROBLEM_GENERATOR_SCRIPT" 4 "$num_blocks" "$SEED" > "$PDDL_FILE" 2>/dev/null
  if [[ $? -ne 0 || ! -s "$PDDL_FILE" ]]; then
    echo "  ERROR: PDDL generation failed for $PROBLEM_BASENAME"
    echo "FG" >> "${RAW_BLOCK_DIR}/.fail_gen"
    rm -f "$PDDL_FILE"
    return 10
  fi

  # 2) Fast Downward
  timeout "$FD_TIMEOUT" "$FD_PATH" --plan-file "$PLAN_FILE" "$DOMAIN_FILE" "$PDDL_FILE" --search "$FD_SEARCH_CONFIG" > /dev/null 2>&1
  if [[ ! -s "$PLAN_FILE" ]]; then
    echo "  WARNING: FD failed/timed out for $PROBLEM_BASENAME"
    echo "FFD" >> "${RAW_BLOCK_DIR}/.fail_fd"
    rm -f "$PLAN_FILE" "$PDDL_FILE"
    return 20
  fi
  local PLAN_LENGTH
  PLAN_LENGTH=$(wc -l < "$PLAN_FILE")
  if [[ "$PLAN_LENGTH" -eq 1 ]]; then
    echo "  WARNING: FD produced empty plan for $PROBLEM_BASENAME"
    echo "FFD" >> "${RAW_BLOCK_DIR}/.fail_fd"
    rm -f "$PLAN_FILE" "$PDDL_FILE"
    return 21
  fi

  # 3) VAL
  "$VALIDATE_PATH" -v "$DOMAIN_FILE" "$PDDL_FILE" "$PLAN_FILE" > "$VAL_OUTPUT_FILE" 2>&1
  if ! grep -q -E "Plan valid|Plan executed successfully" "$VAL_OUTPUT_FILE"; then
    echo "  WARNING: VAL invalid for $PROBLEM_BASENAME"
    echo "FVAL" >> "${RAW_BLOCK_DIR}/.fail_val"
    return 30
  fi

  # 4) Parse & encode
  python "$PARSER_ENCODER_SCRIPT" \
      --val_output_file "$VAL_OUTPUT_FILE" \
      --pddl_domain_file "$DOMAIN_FILE" \
      --pddl_problem_file "$PDDL_FILE" \
      --num_blocks "$num_blocks" \
      --encoding_type "$ENCODING_TYPE" \
      --text_trajectory_output "$TEXT_TRAJECTORY_FILE" \
      --binary_output_prefix "$BINARY_TRAJECTORY_FILE_PREFIX" \
      --raw_data_dir "$RAW_BLOCK_DIR"
  if [[ $? -ne 0 ]]; then
    echo "  ERROR: Parsing/Encoding failed for $PROBLEM_BASENAME"
    echo "FENC" >> "${RAW_BLOCK_DIR}/.fail_enc"
    return 40
  fi

  # 5) Uniqueness check
  local PROBLEM_HASH
  PROBLEM_HASH=$(python "$GET_PROBLEM_HASH_SCRIPT" "$TEXT_TRAJECTORY_FILE" 2>/dev/null || echo "")
  if [[ -z "$PROBLEM_HASH" ]]; then
    echo "  ERROR: Problem hash failed for $PROBLEM_BASENAME"
    echo "FENC" >> "${RAW_BLOCK_DIR}/.fail_enc"
    rm -f "$PDDL_FILE" "$PLAN_FILE" "$VAL_OUTPUT_FILE" "$TEXT_TRAJECTORY_FILE" \
          "${BINARY_TRAJECTORY_FILE_PREFIX}.traj.${ENCODING_TYPE}.npy" \
          "${BINARY_TRAJECTORY_FILE_PREFIX}.goal.${ENCODING_TYPE}.npy"
    return 41
  fi

  # Choose dedup mode
  if [[ "$USE_GLOBAL_DEDUP" -eq 1 ]]; then
    # Global dedup with flock
    touch "$GLOBAL_HASHES_FILE"
    if command -v flock >/dev/null 2>&1; then
      (
        flock -x 9
        if grep -q "$PROBLEM_HASH" "$GLOBAL_HASHES_FILE"; then
          echo "  DUPLICATE(global): $PROBLEM_BASENAME (hash $PROBLEM_HASH)"
          echo "DUP" >> "${RAW_BLOCK_DIR}/.dups"
          rm -f "$PDDL_FILE" "$PLAN_FILE" "$VAL_OUTPUT_FILE" "$TEXT_TRAJECTORY_FILE" \
                "${BINARY_TRAJECTORY_FILE_PREFIX}.traj.${ENCODING_TYPE}.npy" \
                "${BINARY_TRAJECTORY_FILE_PREFIX}.goal.${ENCODING_TYPE}.npy"
          exit 50
        else
          echo "$PROBLEM_HASH" >> "$GLOBAL_HASHES_FILE"
        fi
      ) 9<>"$GLOBAL_HASHES_FILE" || return 99
    else
      # Fallback: lockless (tiny race chance)
      if grep -q "$PROBLEM_HASH" "$GLOBAL_HASHES_FILE"; then
        echo "  DUPLICATE(global-fallback): $PROBLEM_BASENAME"
    echo "DUP" >> "${RAW_BLOCK_DIR}/.dups"
    rm -f "$PDDL_FILE" "$PLAN_FILE" "$VAL_OUTPUT_FILE" "$TEXT_TRAJECTORY_FILE" \
          "${BINARY_TRAJECTORY_FILE_PREFIX}.traj.${ENCODING_TYPE}.npy" \
          "${BINARY_TRAJECTORY_FILE_PREFIX}.goal.${ENCODING_TYPE}.npy"
    return 50
  else
        echo "$PROBLEM_HASH" >> "$GLOBAL_HASHES_FILE"
      fi
    fi
  else
    # Per-shard uniqueness only
    touch "$SHARD_HASHES_FILE"
    if grep -q "$PROBLEM_HASH" "$SHARD_HASHES_FILE"; then
      echo "  DUPLICATE(shard): $PROBLEM_BASENAME (hash $PROBLEM_HASH)"
      echo "DUP" >> "${RAW_BLOCK_DIR}/.dups"
      rm -f "$PDDL_FILE" "$PLAN_FILE" "$VAL_OUTPUT_FILE" "$TEXT_TRAJECTORY_FILE" \
            "${BINARY_TRAJECTORY_FILE_PREFIX}.traj.${ENCODING_TYPE}.npy" \
            "${BINARY_TRAJECTORY_FILE_PREFIX}.goal.${ENCODING_TYPE}.npy"
      return 50
    else
      echo "$PROBLEM_HASH" >> "$SHARD_HASHES_FILE"
    fi
  fi

    echo "  SUCCESS: $PROBLEM_BASENAME"
    echo "OK" >> "${RAW_BLOCK_DIR}/.ok"
    return 0
}

export -f process_problem
export TASK_ID
export USE_GLOBAL_DEDUP

## Main loop over block sizes
for num_blocks in $(seq "$MIN_BLOCKS_TO_GENERATE" "$MAX_BLOCKS_TO_GENERATE"); do
  RAW_BLOCK_DIR="${RAW_PROBLEMS_ROOT}/N${num_blocks}"
  PROCESSED_BLOCK_ENCODING_DIR="${PROCESSED_TRAJECTORIES_ROOT}/N${num_blocks}/${ENCODING_TYPE}"

  mkdir -p "$RAW_BLOCK_DIR/pddl" "$RAW_BLOCK_DIR/plans" "$RAW_BLOCK_DIR/val_out" \
           "$RAW_BLOCK_DIR/trajectories_text" "$RAW_BLOCK_DIR/splits" "$PROCESSED_BLOCK_ENCODING_DIR"

  if [[ -d "$PROCESSED_BLOCK_ENCODING_DIR" && "$OVERWRITE_EXISTING" -eq 1 ]]; then
    echo "Cleaning processed outputs under $PROCESSED_BLOCK_ENCODING_DIR"
    find "$PROCESSED_BLOCK_ENCODING_DIR" -type f -name "blocks_${num_blocks}_problem_*.npy" -delete || true
  elif [[ -d "$PROCESSED_BLOCK_ENCODING_DIR" && "$OVERWRITE_EXISTING" -eq 0 ]]; then
    echo "Processed dir exists; OVERWRITE_EXISTING=0 → skipping N=${num_blocks}"
    continue
  fi

  echo "=== N=${num_blocks}: problems [${I_START}..${I_END}] ==="

  # Prefer GNU parallel; fallback to xargs
  if command -v parallel >/dev/null 2>&1; then
    seq "$I_START" "$I_END" | parallel -j "$PARALLEL_JOBS" --halt soon,fail=1 process_problem "$num_blocks" {}
    RET=$?
  else
    seq "$I_START" "$I_END" | xargs -I{} -P "$PARALLEL_JOBS" bash -c 'process_problem "$0" "$1"' "$num_blocks" {}
    RET=$?
  fi

  # Summarize this shard for this N
  OK_COUNT=$(wc -l < "${RAW_BLOCK_DIR}/.ok"       2>/dev/null || echo 0)
  FG_COUNT=$(wc -l < "${RAW_BLOCK_DIR}/.fail_gen" 2>/dev/null || echo 0)
  FFD_COUNT=$(wc -l < "${RAW_BLOCK_DIR}/.fail_fd" 2>/dev/null || echo 0)
  FVAL_COUNT=$(wc -l < "${RAW_BLOCK_DIR}/.fail_val" 2>/dev/null || echo 0)
  FENC_COUNT=$(wc -l < "${RAW_BLOCK_DIR}/.fail_enc" 2>/dev/null || echo 0)
  DUP_COUNT=$(wc -l < "${RAW_BLOCK_DIR}/.dups"     2>/dev/null || echo 0)

  echo "Shard summary N=${num_blocks}: OK=$OK_COUNT FG=$FG_COUNT FFD=$FFD_COUNT FVAL=$FVAL_COUNT FENC=$FENC_COUNT DUP=$DUP_COUNT"

  TOTAL_SUCCESSFUL=$((TOTAL_SUCCESSFUL + OK_COUNT))
  TOTAL_FAILED_GENERATION=$((TOTAL_FAILED_GENERATION + FG_COUNT))
  TOTAL_FAILED_FD=$((TOTAL_FAILED_FD + FFD_COUNT))
  TOTAL_FAILED_VAL=$((TOTAL_FAILED_VAL + FVAL_COUNT))
  TOTAL_FAILED_ENCODING=$((TOTAL_FAILED_ENCODING + FENC_COUNT))
  TOTAL_DUPLICATES_FILTERED=$((TOTAL_DUPLICATES_FILTERED + DUP_COUNT))

  # Run analysis once after the final shard completes
  if [[ "${TASK_ID}" == "${TOTAL_SHARDS}" ]]; then
    echo "Running analyze_dataset_splits.py for N=${num_blocks} (final shard)"
    python "$ANALYZE_AND_SPLIT_SCRIPT" "$RAW_BLOCK_DIR" || {
      echo "ERROR: Dataset analysis failed for N=${num_blocks}"
    }
  fi
done

## Final summary for this shard
echo "===================================="
echo "Shard $TASK_ID Summary:"
echo "  Successful:                   $TOTAL_SUCCESSFUL"
echo "  Failed PDDL Generation:       $TOTAL_FAILED_GENERATION"
echo "  Failed Fast Downward:         $TOTAL_FAILED_FD"
echo "  Failed VAL Validation:        $TOTAL_FAILED_VAL"
echo "  Failed Encoding:              $TOTAL_FAILED_ENCODING"
echo "  Duplicate Problems Filtered:  $TOTAL_DUPLICATES_FILTERED"
echo "===================================="
