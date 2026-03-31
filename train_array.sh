#!/bin/bash
#SBATCH --job-name=film_patch
#SBATCH --output=slurmouts/film_patch_%A_%a.txt
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-3

ml gcc/12.2.0
ml python/3.10.8
ml llvm/15.0.5
ml cuda/12.6.0
ml cudnn/9.5.0.50-cuda-12.6.0

source ../.venv/bin/activate

ARGS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" train_array_configs.txt)

echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Args: $ARGS"

srun python3 /fred/oz395/DeepJSCC/train.py $ARGS