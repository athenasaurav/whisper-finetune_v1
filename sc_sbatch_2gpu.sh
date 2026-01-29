#!/bin/bash
# 2x A100 80GB. NOTE: finetune.py does NOT use DDP; only 1 GPU is used per process.
# Use this when: (a) you run 2 independent jobs in parallel (each gets 1 GPU), or
# (b) you implement DDP and launch with srun/torchrun.
#SBATCH --job-name=whisper_finetune_${CONFIG_NAME:-job}
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=80:00:00
#SBATCH --partition=a100-80g
#SBATCH --gres=gpu:2
#SBATCH --qos=gpu1week
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=your@email

eval "$(conda shell.bash hook)"
conda activate whisper_finetune
export $(cat .env | xargs)

python src/whisper_finetune/scripts/finetune.py --config "$1"
