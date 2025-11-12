#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# Configuration for runs (kept for clarity; not used programmatically)
# NUM_BLOCKS_ARRAY=(3 4 5 6)
# ENCODINGS_ARRAY=("sas" "bin")
# MODEL_TYPES_ARRAY=("lstm" "ttm" "xgboost")
# DOMAIN_NAME=("blocksworld" "grippers")

echo "Starting submission of PaTS jobs..."

# Create the logs directory if it doesn't exist
mkdir -p logs

# sbatch pats.sh "grippers" "sas" "ttm"
# sleep 1
# sbatch pats.sh "grippers" "bin" "ttm"
# sleep 1
# sbatch pats.sh "grippers" "sas" "xgboost"
# sleep 1
# sbatch pats.sh "grippers" "bin" "xgboost"
# sleep 1
# sbatch pats.sh "grippers" "sas" "lstm"
# sleep 1
# sbatch pats.sh "grippers" "bin" "lstm"
# sleep 1

sbatch pats.sh "blocksworld" "sas" "ttm" 3
sleep 1
sbatch pats.sh "blocksworld" "bin" "ttm" 3
sleep 1
sbatch pats.sh "blocksworld" "sas" "xgboost" 3
sleep 1
sbatch pats.sh "blocksworld" "bin" "xgboost" 3
sleep 1
sbatch pats.sh "blocksworld" "sas" "lstm" 3
sleep 1
sbatch pats.sh "blocksworld" "bin" "lstm" 3
sleep 1

# sbatch pats.sh "blocksworld" "sas" "ttm" 4
# sleep 1
# sbatch pats.sh "blocksworld" "bin" "ttm" 4
# sleep 1
# sbatch pats.sh "blocksworld" "sas" "xgboost" 4
# sleep 1
# sbatch pats.sh "blocksworld" "bin" "xgboost" 4
# sleep 1
# sbatch pats.sh "blocksworld" "sas" "lstm" 4
# sleep 1
# sbatch pats.sh "blocksworld" "bin" "lstm" 4
# sleep 1

# sbatch pats.sh "blocksworld" "sas" "ttm" 5
# sleep 1
# sbatch pats.sh "blocksworld" "bin" "ttm" 5
# sleep 1
# sbatch pats.sh "blocksworld" "sas" "xgboost" 5
# sleep 1
# sbatch pats.sh "blocksworld" "bin" "xgboost" 5
# sleep 1
# sbatch pats.sh "blocksworld" "sas" "lstm" 5
# sleep 1
# sbatch pats.sh "blocksworld" "bin" "lstm" 5
# sleep 1

# sbatch pats.sh "blocksworld" "sas" "ttm" 6
# sleep 1
# sbatch pats.sh "blocksworld" "bin" "ttm" 6
# sleep 1
# sbatch pats.sh "blocksworld" "sas" "xgboost" 6
# sleep 1
# sbatch pats.sh "blocksworld" "bin" "xgboost" 6
# sleep 1
# sbatch pats.sh "blocksworld" "sas" "lstm" 6
# sleep 1
# sbatch pats.sh "blocksworld" "bin" "lstm" 6
# sleep 1

echo "All PaTS jobs submitted."
