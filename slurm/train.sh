#!/usr/bin/env bash
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=job.out

# module load conda
# module load cuda/12.5.82
source activate ml-env
python src/train.py train_small_single_config.yaml