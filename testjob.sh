#!/bin/bash
#
#SBATCH --job-name=gpu_test
#SBATCH --output=test_%j.txt
#
#SBATCH --ntasks=1
#SBATCH --time=40:00
#SBATCH --mem=10000
#SBATCH --gres=gpu:1

ml gcc/12.2.0
ml python/3.10.8
ml llvm/15.0.5
ml cuda/12.6.0
ml cudnn/9.5.0.50-cuda-12.6.0

source ../venv/bin/activate

srun hostname
srun nvidia-smi
srun python3 /fred/oz395/DeepJSCC/train.py
srun sleep 60
