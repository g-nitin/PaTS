#!/bin/bash

# Configuration for runs
# Define the number of blocks to test
NUM_BLOCKS_ARRAY=(3 4 5)

# Define the encoding types to test
ENCODINGS_ARRAY=("sas" "bin")

# Define the model types to test
MODEL_TYPES_ARRAY=("lstm" "ttm" "xgboost" "llama")

# Define the domain(s)
DOMAIN_NAME=("blocksworld")

# Script Logic
echo "Starting submission of PaTS jobs..."

# Create the logs directory if it doesn't exist
mkdir -p logs

sbatch pats.sh 4 "sas" "ttm" "blocksworld"
sleep 1

sbatch pats.sh 4 "bin" "ttm" "blocksworld"
sleep 1

sbatch pats.sh 5 "sas" "ttm" "blocksworld"
sleep 1

sbatch pats.sh 5 "bin" "ttm" "blocksworld"
sleep 1

echo "All PaTS jobs submitted."
