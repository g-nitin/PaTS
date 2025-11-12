domain_name=$1
encoding=$2
model_type=$3
blocksworld_num_blocks=$4

GRIPPERS_ROBOTS=2
GRIPPERS_OBJECTS=3
GRIPPERS_ROOMS=4

if [ "$domain_name" = 'blocksworld' ]; then
    PROBLEM_CONFIG_NAME="N${blocksworld_num_blocks}"
    DOMAIN_ARGS="--num_blocks ${blocksworld_num_blocks}"
elif [ "$domain_name" = 'grippers' ]; then
    PROBLEM_CONFIG_NAME="R${GRIPPERS_ROBOTS}-O${GRIPPERS_OBJECTS}-RM${GRIPPERS_ROOMS}"
    DOMAIN_ARGS="--num-robots ${GRIPPERS_ROBOTS} --num-objects ${GRIPPERS_OBJECTS} --num-rooms ${GRIPPERS_ROOMS}"
else
    echo "Unsupported domain: $domain_name"
    exit 1
fi

echo "Running with:"
echo "  Domain: $domain_name"
echo "  Encoding: $encoding"
echo "  Model Type: $model_type"
echo "  Domain Args: $DOMAIN_ARGS"
echo ""

# Generate timestamp once at the beginning of the script
timestamp=$(date +%Y%m%d_%H%M%S)

# Define base directories for raw and processed data
RAW_PROBLEM_DIR="data/raw_problems/${domain_name}/${PROBLEM_CONFIG_NAME}"
PROCESSED_ENCODING_DIR="data/processed_trajectories/${domain_name}/${PROBLEM_CONFIG_NAME}/${encoding}"

# Define run-specific output directories (grouped by timestamp)
output_base_dir="./training_outputs/run_${timestamp}/${model_type}_${PROBLEM_CONFIG_NAME}_${encoding}"
benchmark_output_dir="./benchmark_results/run_${timestamp}/${model_type}_${PROBLEM_CONFIG_NAME}_${encoding}"

# Create these directories
mkdir -p "${output_base_dir}"
mkdir -p "${benchmark_output_dir}"

# Define log file paths
train_log_file="${output_base_dir}/train_${model_type}_${PROBLEM_CONFIG_NAME}_${encoding}.log"
benchmark_log_file="${benchmark_output_dir}/benchmark_${model_type}_${PROBLEM_CONFIG_NAME}_${encoding}.log"

# Assuming MIN_BLOCKS_TO_GENERATE and MAX_BLOCKS_TO_GENERATE are typically the same
# or that the script is run for a single num_blocks at a time.
# The 'num_blocks' variable will hold the last value from the loop.
MAX_PLAN_LENGTH_FILE="${RAW_PROBLEM_DIR}/splits/max_plan_length.txt"
dynamic_max_plan_length=60 # Default fallback value
if [ -f "$MAX_PLAN_LENGTH_FILE" ]; then
    read -r dynamic_max_plan_length <"$MAX_PLAN_LENGTH_FILE"
    echo "Dynamically determined max plan length: $dynamic_max_plan_length"
else
    echo "Warning: max_plan_length.txt not found at ${MAX_PLAN_LENGTH_FILE}. Using default max_plan_length: 60." >&2
    exit 1
fi

if [ "$model_type" = 'lstm' ]; then
    echo "Using LSTM model"
    echo ""

    model_path="${output_base_dir}/${model_type}_${PROBLEM_CONFIG_NAME}/pats_lstm_model_${PROBLEM_CONFIG_NAME}.pth"
    uv run python -m scripts.train_model \
        --model_type $model_type \
        --dataset_dir "$RAW_PROBLEM_DIR" \
        --processed_encoding_dir "$PROCESSED_ENCODING_DIR" \
        --domain "$domain_name" \
        $DOMAIN_ARGS \
        --output_dir "$output_base_dir" \
        --encoding_type "$encoding" \
        --epochs 200 \
        --batch_size 32 \
        --learning_rate 0.001 \
        --seed 13 \
        >"${train_log_file}" 2>&1 # Redirect stdout and stderr to log file

    echo ""
    echo "Training completed. Outputs in ${output_base_dir}. Log in ${train_log_file}."
    echo "Starting benchmarking with model: $model_type. Log in ${benchmark_log_file}."
    echo ""

    uv run python -m scripts.benchmark \
        --dataset_dir "$RAW_PROBLEM_DIR" \
        --processed_encoding_dir "$PROCESSED_ENCODING_DIR" \
        --domain "$domain_name" \
        $DOMAIN_ARGS \
        --model_type "$model_type" \
        --model_path "$model_path" \
        --output_dir "$benchmark_output_dir" \
        --encoding_type "$encoding" \
        --max_plan_length "$dynamic_max_plan_length" \
        --save_detailed_results \
        >"${benchmark_log_file}" 2>&1 # Redirect stdout and stderr to log file

    echo "Benchmarking completed. Results in ${benchmark_output_dir}. Log in ${benchmark_log_file}."

elif [ "$model_type" = 'ttm' ]; then
    echo "Using TTM model"
    model_path="${output_base_dir}/${model_type}_${PROBLEM_CONFIG_NAME}/final_model_assets"
    echo ""

    uv run python -m scripts.train_model \
        --model_type $model_type \
        --dataset_dir "$RAW_PROBLEM_DIR" \
        --processed_encoding_dir "$PROCESSED_ENCODING_DIR" \
        --domain "$domain_name" \
        $DOMAIN_ARGS \
        --output_dir "${output_base_dir}" \
        --encoding_type "$encoding" \
        --epochs 200 \
        --batch_size 32 \
        --learning_rate 0.001 \
        --seed 13 \
        >"${train_log_file}" 2>&1 # Redirect stdout and stderr to log file

    echo ""
    echo "Training completed. Outputs in ${output_base_dir}. Log in ${train_log_file}."
    echo "Starting benchmarking with model: $model_type. Log: ${benchmark_log_file}."
    echo ""

    uv run python -m scripts.benchmark \
        --dataset_dir "$RAW_PROBLEM_DIR" \
        --processed_encoding_dir "$PROCESSED_ENCODING_DIR" \
        --domain "$domain_name" \
        $DOMAIN_ARGS \
        --model_type "$model_type" \
        --model_path "$model_path" \
        --output_dir "$benchmark_output_dir" \
        --encoding_type "$encoding" \
        --max_plan_length "$dynamic_max_plan_length" \
        --save_detailed_results \
        >"${benchmark_log_file}" 2>&1

    echo "Benchmarking completed. Results in ${benchmark_output_dir}. Log in ${benchmark_log_file}."

elif [ "$model_type" = 'xgboost' ]; then
    echo "Using XGBoost model"
    model_path="${output_base_dir}/${model_type}_${PROBLEM_CONFIG_NAME}/pats_xgboost_model_${PROBLEM_CONFIG_NAME}.joblib"
    echo ""

    uv run python -m scripts.train_model \
        --model_type $model_type \
        --dataset_dir "$RAW_PROBLEM_DIR" \
        --processed_encoding_dir "$PROCESSED_ENCODING_DIR" \
        --domain "$domain_name" \
        $DOMAIN_ARGS \
        --output_dir "${output_base_dir}" \
        --encoding_type "$encoding" \
        --epochs 200 \
        --batch_size 32 \
        --learning_rate 0.001 \
        --seed 13 \
        --xgboost_context_window_size 3 \
        >"${train_log_file}" 2>&1

    echo ""
    echo "Training completed. Outputs in ${output_base_dir}. Log in ${train_log_file}."
    echo "Starting benchmarking with model: $model_type. Log: ${benchmark_log_file}."
    echo ""
    uv run python -m scripts.benchmark \
        --dataset_dir "$RAW_PROBLEM_DIR" \
        --processed_encoding_dir "$PROCESSED_ENCODING_DIR" \
        --domain "$domain_name" \
        $DOMAIN_ARGS \
        --model_type "$model_type" \
        --model_path "$model_path" \
        --output_dir "$benchmark_output_dir" \
        --encoding_type "$encoding" \
        --max_plan_length "$dynamic_max_plan_length" \
        --save_detailed_results \
        --xgboost_context_window_size 3 \
        >"${benchmark_log_file}" 2>&1

    echo "Benchmarking completed. Results in ${benchmark_output_dir}. Log in ${benchmark_log_file}."

elif [ "$model_type" = 'llama' ]; then
    echo "Using Llama model"

    # For Llama, model_path will be the model_id directly, as it's not a local file path
    model_path="meta-llama/Llama-3.1-8B-Instruct"

    # 1. Benchmark Zero-shot Llama
    echo ""
    echo "Starting benchmarking with Zero-shot Llama."
    # Define specific output and log paths for zero-shot
    zero_shot_benchmark_output_dir="${benchmark_output_dir}_zero_shot" # This will be a sibling to the main benchmark_output_dir
    mkdir -p "$zero_shot_benchmark_output_dir"
    zero_shot_benchmark_log_file="${zero_shot_benchmark_output_dir}/benchmark_${model_type}_${PROBLEM_CONFIG_NAME}_${encoding}_zero_shot.log"
    echo "Log: ${zero_shot_benchmark_log_file}."

    # --llama_use_few_shot is NOT present for zero-shot
    uv run python -m scripts.benchmark \
        --dataset_dir "$RAW_PROBLEM_DIR" \
        --processed_encoding_dir "$PROCESSED_ENCODING_DIR" \
        --domain "$domain_name" \
        $DOMAIN_ARGS \
        --model_type "$model_type" \
        --model_path "$model_path" \
        --output_dir "$zero_shot_benchmark_output_dir" \
        --encoding_type "$encoding" \
        --max_plan_length "$dynamic_max_plan_length" \
        --save_detailed_results \
        >"${zero_shot_benchmark_log_file}" 2>&1

    echo "Zero-shot Llama benchmarking completed. Results in ${zero_shot_benchmark_output_dir}."

    # 2. Benchmark Few-shot Llama
    echo ""
    echo "Starting benchmarking with Few-shot Llama."
    # Define specific output and log paths for few-shot
    few_shot_benchmark_output_dir="${benchmark_output_dir}_few_shot" # This will be a sibling to the main benchmark_output_dir
    mkdir -p "$few_shot_benchmark_output_dir"
    few_shot_benchmark_log_file="${few_shot_benchmark_output_dir}/benchmark_${model_type}_${PROBLEM_CONFIG_NAME}_${encoding}_few_shot.log"
    echo "Log: ${few_shot_benchmark_log_file}."

    uv run python -m scripts.benchmark \
        --dataset_dir "$RAW_PROBLEM_DIR" \
        --processed_encoding_dir "$PROCESSED_ENCODING_DIR" \
        --domain "$domain_name" \
        $DOMAIN_ARGS \
        --model_type "$model_type" \
        --model_path "$model_path" \
        --output_dir "$few_shot_benchmark_output_dir" \
        --encoding_type "$encoding" \
        --max_plan_length "$dynamic_max_plan_length" \
        --save_detailed_results \
        --llama_use_few_shot \
        >"${few_shot_benchmark_log_file}" 2>&1

    echo "Few-shot Llama benchmarking completed. Results in ${few_shot_benchmark_output_dir}."

    echo ""
    echo "All benchmarking for $model_type completed."

else
    echo "Unsupported model type: $model_type"
    exit 1
fi
