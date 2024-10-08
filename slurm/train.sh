#!/usr/bin/env bash
#SBATCH --gpus=T4:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=job.out

# module load conda
# module load cuda/12.5.82
source activate ml-env
python src/train.py train_ic.yaml