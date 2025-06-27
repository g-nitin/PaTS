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

model_type='lstm'
num_blocks=4

# Generate timestamp and build unique dirs/paths
timestamp=$(date +%Y%m%d_%H%M%S)
output_dir="./training_outputs/${model_type}_${timestamp}"
benchmark_output_dir="./benchmark_results/${model_type}_${timestamp}"
model_path="${output_dir}/${model_type}_N${num_blocks}/pats_${model_type}_model_N${num_blocks}.pth"

echo -e "\n"
echo "Starting training with model: $model_type"
echo "Using masking"
python -m scripts.train_model \
    --model_type $model_type \
    --dataset_dir data/blocks_4 \
    --dataset_split_dir data/blocks_4 \
    --num_blocks $num_blocks \
    --output_dir $output_dir \
    --epochs 400 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --seed 13 \
    --use_mlm_task \
    --mlm_loss_weight 0.2 \
    --mlm_mask_prob 0.15

echo -e "\n"
echo "Training completed. Outputs in $output_dir"

echo -e "\n"
echo "Starting benchmarking with model: $model_type"
python -m scripts.benchmark \
    --dataset_dir ./data \
    --num_blocks $num_blocks \
    --model_type $model_type \
    --model_path $model_path \
    --output_dir $benchmark_output_dir \
    --max_plan_length 60 \
    --save_detailed_results

echo -e "\n"
echo "Benchmarking completed. Results in $benchmark_output_dir"