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

model_type='llama'  # `lstm`, `ttm`, `xgboost`, `llama`
num_blocks=4
encoding='sas'  # `sas`, `bin`
dataset_dir="data/blocks_${num_blocks}-${encoding}"

# Generate timestamp and build unique dirs/paths
timestamp=$(date +%Y%m%d_%H%M%S)
output_base_dir="./training_outputs_${encoding}/${model_type}_${timestamp}"
benchmark_output_dir="./benchmark_results_${encoding}/${model_type}_${timestamp}"
llama_model_id="meta-llama/Llama-3.1-8B-Instruct"

mkdir -p "${output_base_dir}"


if [ "$model_type" = 'lstm' ]; then
    echo "Using LSTM model"
    model_path="${output_base_dir}/${model_type}_N${num_blocks}/pats_lstm_model_N${num_blocks}.pth"
    echo -e "\n"
    python -m scripts.train_model \
        --model_type $model_type \
        --dataset_dir "$dataset_dir" \
        --dataset_split_dir "$dataset_dir" \
        --num_blocks "$num_blocks" \
        --encoding_type "$encoding" \
        --output_dir "${output_base_dir}" \
        --epochs 400 \
        --batch_size 32 \
        --learning_rate 0.001 \
        --seed 13 \
        # LSTM specific args here if needed (e.g., --use_mlm_task, --use_constraint_loss)
    echo -e "\n"
    echo "Training completed. Outputs in ${output_base_dir}"
elif [ "$model_type" = 'ttm' ]; then
    echo "Using TTM model"
    model_path="${output_base_dir}/${model_type}_N${num_blocks}/final_model_assets"
    echo -e "\n"
    python -m scripts.train_model \
        --model_type $model_type \
        --dataset_dir "$dataset_dir" \
        --dataset_split_dir "$dataset_dir" \
        --num_blocks "$num_blocks" \
        --encoding_type "$encoding" \
        --output_dir "${output_base_dir}" \
        --epochs 400 \
        --batch_size 32 \
        --learning_rate 0.001 \
        --seed 13 \
        # TTM specific args here if needed
elif [ "$model_type" = 'xgboost' ]; then
    echo "Using XGBoost model"
    model_path="${output_base_dir}/${model_type}_N${num_blocks}/pats_xgboost_model_N${num_blocks}.joblib"
    echo -e "\n"
    python -m scripts.train_model \
        --model_type $model_type \
        --dataset_dir "$dataset_dir" \
        --dataset_split_dir "$dataset_dir" \
        --num_blocks "$num_blocks" \
        --encoding_type "$encoding" \
        --output_dir "${output_base_dir}" \
        --epochs 400 \
        --batch_size 32 \
        --learning_rate 0.001 \
        --seed 13 \
        --xgboost_context_window_size 3 # Ensure this matches what was used for training
    echo -e "\n"
    echo "Training completed. Outputs in ${output_base_dir}"
elif [ "$model_type" = 'llama' ]; then
    echo "Using Llama model"
    # For Llama, model_path will be the model_id directly, as it's not a local file path
    model_path="$llama_model_id"
    # No "training" step for Llama, directly proceed to benchmarking.
else
    echo "Unsupported model type: $model_type"
    exit 1
fi

mkdir -p "$benchmark_output_dir" # Ensure benchmark output directory exists

echo -e "\n"
echo "Starting benchmarking with model: $model_type"
python -m scripts.benchmark \
    --dataset_dir "$dataset_dir" \
    --num_blocks "$num_blocks" \
    --model_type "$model_type" \
    --model_path "$model_path" \
    --output_dir "$benchmark_output_dir" \
    --encoding_type "$encoding" \
    --max_plan_length 60 \
    --save_detailed_results \
    --xgboost_context_window_size 3 # Keep for other models, Llama will ignore

echo -e "\n"
echo "Benchmarking completed. Results in $benchmark_output_dir"