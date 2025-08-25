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

model_type='llama_finetune'  # `lstm`, `ttm`, `xgboost`, `llama`, `llama_finetune`
num_blocks=4
encoding='sas'  # `sas`, `bin`
dataset_dir="data/blocks_${num_blocks}-${encoding}"

# Generate timestamp and build unique dirs/paths
timestamp=$(date +%Y%m%d_%H%M%S)
output_base_dir="./training_outputs/training_outputs_${encoding}/${model_type}_${timestamp}"
benchmark_output_dir="./benchmark_results/benchmark_results_${encoding}/${model_type}_${timestamp}"
llama_model_id="meta-llama/Llama-3.1-8B-Instruct"

mkdir -p "${output_base_dir}"
mkdir -p "$benchmark_output_dir" # Ensure benchmark output directory exists

if [ "$model_type" = 'lstm' ]; then
    echo "Using LSTM model"
    model_path="${output_base_dir}/${model_type}_N${num_blocks}/pats_lstm_model_N${num_blocks}.pth"
    echo "\n"
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
    echo "\n"
    echo "Training completed. Outputs in ${output_base_dir}"

    echo "\n"
    echo "Starting benchmarking with model: $model_type"
    python -m scripts.benchmark \
        --dataset_dir "$dataset_dir" \
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

    echo "\n"
    echo "Starting benchmarking with model: $model_type"
    python -m scripts.benchmark \
        --dataset_dir "$dataset_dir" \
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
    echo "\n"
    echo "Training completed. Outputs in ${output_base_dir}"

    echo "\n"
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
        --xgboost_context_window_size 3

    echo "\n"
    echo "Benchmarking completed. Results in $benchmark_output_dir"

elif [ "$model_type" = 'llama' ]; then   # This block is for base Llama (zero-shot/few-shot)
    echo "Using Llama model (base, no fine-tuning)"
    model_path="$llama_model_id" # For Llama, model_path is the model_id directly

    # 1. Benchmark Zero-shot Llama 
    echo "\n"
    echo "Starting benchmarking with Zero-shot Llama (base model)"
    zero_shot_benchmark_output_dir="${benchmark_output_dir}_zero_shot"
    mkdir -p "$zero_shot_benchmark_output_dir"

    python -m scripts.benchmark \
        --dataset_dir "$dataset_dir" \
        --num_blocks "$num_blocks" \
        --model_type "$model_type" \
        --model_path "$model_path" \
        --output_dir "$zero_shot_benchmark_output_dir" \
        --encoding_type "$encoding" \
        --max_plan_length 60 \
        --save_detailed_results \
        --xgboost_context_window_size 3 # Llama will ignore this
        --llama_model_id "$llama_model_id"

    echo "Zero-shot Llama benchmarking completed. Results in $zero_shot_benchmark_output_dir"

    # 2. Benchmark Few-shot Llama 
    echo "\n"
    echo "Starting benchmarking with Few-shot Llama (base model)"
    few_shot_benchmark_output_dir="${benchmark_output_dir}_few_shot"
    mkdir -p "$few_shot_benchmark_output_dir"

    python -m scripts.benchmark \
        --dataset_dir "$dataset_dir" \
        --num_blocks "$num_blocks" \
        --model_type "$model_type" \
        --model_path "$model_path" \
        --output_dir "$few_shot_benchmark_output_dir" \
        --encoding_type "$encoding" \
        --max_plan_length 60 \
        --save_detailed_results \
        --xgboost_context_window_size 3 \
        --llama_use_few_shot \
        --llama_model_id "$llama_model_id"

    echo "Few-shot Llama benchmarking completed. Results in $few_shot_benchmark_output_dir"

    echo "\n"
    echo "All benchmarking for base $model_type completed."

elif [ "$model_type" = 'llama_finetune' ]; then # This block is for Llama fine-tuning and subsequent benchmarking
    echo "Using Llama for fine-tuning"
    base_llama_model_id="$llama_model_id" # Use the original model_id for base model loading
    # Define paths for fine-tuned Llama
    llama_finetuned_adapter_path="${output_base_dir}/llama_finetune_N${num_blocks}/lora_adapter"

    echo "\n"
    echo "Starting Llama LoRA fine-tuning..."
    python -m scripts.train_model \
        --model_type "$model_type" \
        --dataset_dir "$dataset_dir" \
        --dataset_split_dir "$dataset_dir" \
        --num_blocks "$num_blocks" \
        --encoding_type "$encoding" \
        --output_dir "${output_base_dir}" \
        --epochs 3 \
        --batch_size 4 \
        --learning_rate 2e-4 \
        --seed 13 \
        --llama_model_id "$base_llama_model_id" \
        --lora_r 16 \
        --lora_alpha 32 \
        --lora_dropout 0.05 \
        --lora_target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj" \
        --llama_max_seq_len 512 \
        --gradient_accumulation_steps 4 \
        --warmup_ratio 0.03 \
        --weight_decay 0.01 \
        --logging_steps 10 \
        --save_steps 50 \
        --eval_steps 50

    echo "\n"
    echo "Llama LoRA fine-tuning completed. Adapter saved to ${llama_finetuned_adapter_path}"

    # 1. Benchmark Fine-tuned Zero-shot Llama
    echo "\n"
    echo "Starting benchmarking with Fine-tuned Zero-shot Llama"
    finetuned_zero_shot_benchmark_output_dir="${benchmark_output_dir}_finetuned_zero_shot"
    mkdir -p "$finetuned_zero_shot_benchmark_output_dir"

    python -m scripts.benchmark \
        --dataset_dir "$dataset_dir" \
        --num_blocks "$num_blocks" \
        --model_type "llama" \
        --model_path "$llama_finetuned_adapter_path" \
        --output_dir "$finetuned_zero_shot_benchmark_output_dir" \
        --encoding_type "$encoding" \
        --max_plan_length 60 \
        --save_detailed_results \
        --xgboost_context_window_size 3 \
        --llama_model_id "$base_llama_model_id" # Pass base model ID for LlamaWrapper to load base model for adapter

    echo "Fine-tuned Zero-shot Llama benchmarking completed. Results in $finetuned_zero_shot_benchmark_output_dir"

    # 2. Benchmark Fine-tuned Few-shot Llama
    echo "\n"
    echo "Starting benchmarking with Fine-tuned Few-shot Llama"
    finetuned_few_shot_benchmark_output_dir="${benchmark_output_dir}_finetuned_few_shot"
    mkdir -p "$finetuned_few_shot_benchmark_output_dir"

    python -m scripts.benchmark \
        --dataset_dir "$dataset_dir" \
        --num_blocks "$num_blocks" \
        --model_type "llama" \
        --model_path "$llama_finetuned_adapter_path" \
        --output_dir "$finetuned_few_shot_benchmark_output_dir" \
        --encoding_type "$encoding" \
        --max_plan_length 60 \
        --save_detailed_results \
        --xgboost_context_window_size 3 \
        --llama_use_few_shot \
        --llama_model_id "$base_llama_model_id" # Pass base model ID for LlamaWrapper to load base model for adapter

    echo "Fine-tuned Few-shot Llama benchmarking completed. Results in $finetuned_few_shot_benchmark_output_dir"

    echo "\n"
    echo "All fine-tuning and benchmarking for $model_type completed."
else
    echo "Unsupported model type: $model_type"
    exit 1
fi
