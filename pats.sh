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

RAW_BLOCK_DIR="data/raw_problems/$${domain_name}/N$${num_blocks}"
PROCESSED_BLOCK_ENCODING_DIR="data/processed_trajectories/$${domain_name}/N$${num_blocks}/${encoding}"

# Generate timestamp and build unique dirs/paths
timestamp=$(date +%Y%m%d_%H%M%S)
output_base_dir="./training_outputs/training_outputs_${encoding}/${model_type}_${timestamp}"
benchmark_output_dir="./benchmark_results/benchmark_results_${encoding}/${model_type}_${timestamp}"
mkdir -p "${output_base_dir}"
mkdir -p "$benchmark_output_dir"

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

    echo "\n"
    echo "Training completed. Outputs in ${output_base_dir}"

    echo "\n"
    echo "Starting benchmarking with model: $model_type"
    python -m scripts.benchmark \
        --dataset_dir "$RAW_BLOCK_DIR" \
        --num_blocks "$num_blocks" \
        --model_type "$model_type" \
        --model_path "$model_path" \
        --output_dir "$benchmark_output_dir" \
        --encoding_type "$encoding" \
        --max_plan_length 60 \
        --save_detailed_results
    
    echo "\n"
    echo "Benchmarking completed. Results in $benchmark_output_dir"

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
        # TTM specific args here if needed

    echo "\n"
    echo "Starting benchmarking with model: $model_type"
    python -m scripts.benchmark \
        --dataset_dir "$RAW_BLOCK_DIR" \
        --num_blocks "$num_blocks" \
        --model_type "$model_type" \
        --model_path "$model_path" \
        --output_dir "$benchmark_output_dir" \
        --encoding_type "$encoding" \
        --max_plan_length 60 \
        --save_detailed_results
    
    echo "\n"
    echo "Benchmarking completed. Results in $benchmark_output_dir"

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
        --xgboost_context_window_size 3 # Ensure this matches what was used for training
    echo "\n"
    echo "Training completed. Outputs in ${output_base_dir}"

    echo "\n"
    echo "Starting benchmarking with model: $model_type"
    python -m scripts.benchmark \
        --dataset_dir "$RAW_BLOCK_DIR" \
        --num_blocks "$num_blocks" \
        --model_type "$model_type" \
        --model_path "$model_path" \
        --output_dir "$benchmark_output_dir" \
        --encoding_type "$encoding" \
        --max_plan_length 60 \
        --save_detailed_results \
        --xgboost_context_window_size 3

    echo "\n"
    echo "Benchmarking completed. Results in $benchmark_output_dir"

elif [ "$model_type" = 'llama' ]; then
    echo "Using Llama model"
    # For Llama, model_path will be the model_id directly, as it's not a local file path
    model_path="$llama_model_id"

    # 1. Benchmark Zero-shot Llama 
    echo "\n"
    echo "Starting benchmarking with Zero-shot Llama"
    zero_shot_benchmark_output_dir="${benchmark_output_dir}_zero_shot"
    mkdir -p "$zero_shot_benchmark_output_dir"

    python -m scripts.benchmark \
        --dataset_dir "$RAW_BLOCK_DIR" \
        --num_blocks "$num_blocks" \
        --model_type "$model_type" \
        --model_path "$model_path" \
        --output_dir "$zero_shot_benchmark_output_dir" \
        --encoding_type "$encoding" \
        --max_plan_length 60 \
        --save_detailed_results \
        --xgboost_context_window_size 3 # Keep for other models, Llama will ignore
        # --llama_use_few_shot is NOT present for zero-shot

    echo "Zero-shot Llama benchmarking completed. Results in $zero_shot_benchmark_output_dir"

    # 2. Benchmark Few-shot Llama 
    echo "\n"
    echo "Starting benchmarking with Few-shot Llama"
    few_shot_benchmark_output_dir="${benchmark_output_dir}_few_shot"
    mkdir -p "$few_shot_benchmark_output_dir"

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
        --llama_use_few_shot # ADD THIS FLAG FOR FEW-SHOT

    echo "Few-shot Llama benchmarking completed. Results in $few_shot_benchmark_output_dir"

    echo "\n"
    echo "All benchmarking for $model_type completed."
else
    echo "Unsupported model type: $model_type"
    exit 1
fi
