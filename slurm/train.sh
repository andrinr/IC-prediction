#!/usr/bin/env bash
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=job.out

module load anaconda3
conda activate ml-env
python src/train.py --config default-config.yaml