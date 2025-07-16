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

model_type='ttm'
num_blocks=4
encoding='bin'  # `sas`, `bin`
dataset_dir="data/blocks_${num_blocks}-${encoding}"

# Generate timestamp and build unique dirs/paths
timestamp='20250716_152958'

output_dir="./training_outputs_${encoding}/${model_type}_${timestamp}"
benchmark_output_dir="./benchmark_results_${encoding}/${model_type}_${timestamp}"

if [ "$model_type" = 'lstm' ]; then
    echo "Using LSTM model"
    model_path="${output_dir}/${model_type}_N${num_blocks}/pats_lstm_model_N${num_blocks}.pth"
elif [ "$model_type" = 'ttm' ]; then
    echo "Using TTM model"
    model_path="${output_dir}/${model_type}_N${num_blocks}/final_model_assets"
else
    echo "Unsupported model type: $model_type"
    exit 1
fi

mkdir -p "$output_dir"
mkdir -p "$benchmark_output_dir"

echo -e "\n"
echo "Starting benchmarking with model: $model_type"
python -m scripts.benchmark \
    --dataset_dir $dataset_dir \
    --num_blocks $num_blocks \
    --model_type $model_type \
    --model_path $model_path \
    --output_dir $benchmark_output_dir \
    --encoding_type $encoding \
    --max_plan_length 60 \
    --save_detailed_results

echo -e "\n"
echo "Benchmarking completed. Results in $benchmark_output_dir"
