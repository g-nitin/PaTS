#! /bin/bash

#SBATCH --job-name=PaTS
#SBATCH -o r_out%j.out
#SBATCH -e r_err%j.err

#SBATCH --mail-user=niting@email.sc.edu
#SBATCH --mail-type=ALL

#SBATCH -p v100-16gb-hiprio
#SBATCH --gres=gpu:1

module load python3/anaconda/2021.07 gcc/12.2.0 cuda/12.1
source activate /home/niting/miniconda3/envs/pats-env

echo $CONDA_DEFAULT_ENV
hostname
echo "Python version: $(python --version)"

num_blocks=4
domain_name="blocksworld"
encoding='sas' # `sas`, `bin`
model_type='lstm'  # `lstm`, `ttm`, `xgboost`, `llama`

# If `model_type` == 'llama'
llama_model_id="meta-llama/Llama-3.1-8B-Instruct"

# Generate timestamp once at the beginning of the script
timestamp=$(date +%Y%m%d_%H%M%S)

# Define base directories for raw and processed data
RAW_BLOCK_DIR="data/raw_problems/${domain_name}/N${num_blocks}"
PROCESSED_BLOCK_ENCODING_DIR="data/processed_trajectories/${domain_name}/N${num_blocks}/${encoding}"

# Define run-specific output directories (grouped by timestamp)
output_base_dir="./training_outputs/run_${timestamp}/${model_type}_N${num_blocks}_${encoding}"
benchmark_output_dir="./benchmark_results/run_${timestamp}/${model_type}_N${num_blocks}_${encoding}"

# Create these directories
mkdir -p "${output_base_dir}"
mkdir -p "${benchmark_output_dir}"

# Define log file paths
train_log_file="${output_base_dir}/train_${model_type}_N${num_blocks}_${encoding}.log"
benchmark_log_file="${benchmark_output_dir}/benchmark_${model_type}_N${num_blocks}_${encoding}.log"

if [ "$model_type" = 'lstm' ]; then
    echo "Using LSTM model"
    echo "\n"
    
    model_path="${output_base_dir}/${model_type}_N${num_blocks}/pats_lstm_model_N${num_blocks}.pth"
    python -m scripts.train_model \
        --model_type $model_type \
        --dataset_dir "$RAW_BLOCK_DIR" \
        --dataset_split_dir "${RAW_BLOCK_DIR}/splits" \
        --processed_data_dir "$PROCESSED_BLOCK_ENCODING_DIR" \
        --num_blocks "$num_blocks" \
        --output_dir "${output_base_dir}" \
        --encoding_type "$encoding" \
        --epochs 400 \
        --batch_size 32 \
        --learning_rate 0.001 \
        --seed 13 \
        # LSTM specific args here following...
        > "${train_log_file}" 2>&1  # Redirect stdout and stderr to log file

    echo "\n"
    echo "Training completed. Outputs in ${output_base_dir}. Log in ${train_log_file}."
    echo "Starting benchmarking with model: $model_type. Log in ${benchmark_log_file}."
    echo "\n"

    python -m scripts.benchmark \
        --dataset_dir "$RAW_BLOCK_DIR" \
        --num_blocks "$num_blocks" \
        --model_type "$model_type" \
        --model_path "$model_path" \
        --output_dir "$benchmark_output_dir" \
        --encoding_type "$encoding" \
        --max_plan_length 60 \
        --save_detailed_results \
        > "${benchmark_log_file}" 2>&1 # Redirect stdout and stderr to log file

    echo "Benchmarking completed. Results in ${benchmark_output_dir}. Log in ${benchmark_log_file}."

elif [ "$model_type" = 'ttm' ]; then
    echo "Using TTM model"
    model_path="${output_base_dir}/${model_type}_N${num_blocks}/final_model_assets"
    echo "\n"
    python -m scripts.train_model \
        --model_type $model_type \
        --dataset_dir "$RAW_BLOCK_DIR" \
        --processed_data_dir "$PROCESSED_BLOCK_ENCODING_DIR" \
        --num_blocks "$num_blocks" \
        --encoding_type "$encoding" \
        --output_dir "${output_base_dir}" \
        --epochs 400 \
        --batch_size 32 \
        --learning_rate 0.001 \
        --seed 13 \
        # TTM specific args here following...
        > "${train_log_file}" 2>&1  # Redirect stdout and stderr to log file

    echo "\n"
    echo "Training completed. Outputs in ${output_base_dir}. Log in ${train_log_file}."
    echo "Starting benchmarking with model: $model_type. Log: ${benchmark_log_file}."
    echo "\n"
    
    python -m scripts.benchmark \
        --dataset_dir "$RAW_BLOCK_DIR" \
        --num_blocks "$num_blocks" \
        --model_type "$model_type" \
        --model_path "$model_path" \
        --output_dir "$benchmark_output_dir" \
        --encoding_type "$encoding" \
        --max_plan_length 60 \
        --save_detailed_results \
        > "${benchmark_log_file}" 2>&1
    
    echo "Benchmarking completed. Results in ${benchmark_output_dir}. Log in ${benchmark_log_file}."

elif [ "$model_type" = 'xgboost' ]; then
    echo "Using XGBoost model"
    model_path="${output_base_dir}/${model_type}_N${num_blocks}/pats_xgboost_model_N${num_blocks}.joblib"
    echo "\n"
    python -m scripts.train_model \
        --model_type $model_type \
        --dataset_dir "$RAW_BLOCK_DIR" \
        --processed_data_dir "$PROCESSED_BLOCK_ENCODING_DIR" \
        --num_blocks "$num_blocks" \
        --encoding_type "$encoding" \
        --output_dir "${output_base_dir}" \
        --epochs 400 \
        --batch_size 32 \
        --learning_rate 0.001 \
        --seed 13 \
        --xgboost_context_window_size 3 \
        > "${train_log_file}" 2>&1
    
    echo "\n"
    echo "Training completed. Outputs in ${output_base_dir}. Log in ${train_log_file}."
    echo "Starting benchmarking with model: $model_type. Log: ${benchmark_log_file}."
    echo "\n"

    python -m scripts.benchmark \
        --dataset_dir "$RAW_BLOCK_DIR" \
        --num_blocks "$num_blocks" \
        --model_type "$model_type" \
        --model_path "$model_path" \
        --output_dir "$benchmark_output_dir" \
        --encoding_type "$encoding" \
        --max_plan_length 60 \
        --save_detailed_results \
        --xgboost_context_window_size 3 \
        > "${benchmark_log_file}" 2>&1
    
    echo "Benchmarking completed. Results in ${benchmark_output_dir}. Log in ${benchmark_log_file}."

elif [ "$model_type" = 'llama' ]; then
    echo "Using Llama model"
    # For Llama, model_path will be the model_id directly, as it's not a local file path
    model_path="$llama_model_id"

    # 1. Benchmark Zero-shot Llama 
    echo "\n"
    echo "Starting benchmarking with Zero-shot Llama."
    # Define specific output and log paths for zero-shot
    zero_shot_benchmark_output_dir="${benchmark_output_dir}_zero_shot" # This will be a sibling to the main benchmark_output_dir
    mkdir -p "$zero_shot_benchmark_output_dir"
    zero_shot_benchmark_log_file="${zero_shot_benchmark_output_dir}/benchmark_${model_type}_N${num_blocks}_${encoding}_zero_shot.log"
    echo "Log: ${zero_shot_benchmark_log_file}."

    # --llama_use_few_shot is NOT present for zero-shot
    python -m scripts.benchmark \
        --dataset_dir "$RAW_BLOCK_DIR" \
        --num_blocks "$num_blocks" \
        --model_type "$model_type" \
        --model_path "$model_path" \
        --output_dir "$zero_shot_benchmark_output_dir" \
        --encoding_type "$encoding" \
        --max_plan_length 60 \
        --save_detailed_results \
        > "${zero_shot_benchmark_log_file}" 2>&1

    echo "Zero-shot Llama benchmarking completed. Results in ${zero_shot_benchmark_output_dir}. Log in ${zero_shot_benchmark_log_file}."

    # 2. Benchmark Few-shot Llama 
    echo "\n"
    echo "Starting benchmarking with Few-shot Llama."
    # Define specific output and log paths for few-shot
    few_shot_benchmark_output_dir="${benchmark_output_dir}_few_shot" # This will be a sibling to the main benchmark_output_dir
    mkdir -p "$few_shot_benchmark_output_dir"
    few_shot_benchmark_log_file="${few_shot_benchmark_output_dir}/benchmark_${model_type}_N${num_blocks}_${encoding}_few_shot.log"
    echo "Log: ${few_shot_benchmark_log_file}."

    python -m scripts.benchmark \
        --dataset_dir "$RAW_BLOCK_DIR" \
        --num_blocks "$num_blocks" \
        --model_type "$model_type" \
        --model_path "$model_path" \
        --output_dir "$few_shot_benchmark_output_dir" \
        --encoding_type "$encoding" \
        --max_plan_length 60 \
        --save_detailed_results \
        --xgboost_context_window_size 3 \
        --llama_use_few_shot \
        > "${few_shot_benchmark_log_file}" 2>&1

    echo "Few-shot Llama benchmarking completed. Results in ${few_shot_benchmark_output_dir}. Log in ${few_shot_benchmark_log_file}"

    echo "\n"
    echo "All benchmarking for $model_type completed."

else
    echo "Unsupported model type: $model_type"
    exit 1
fi
