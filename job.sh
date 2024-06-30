#!/bin/bash -l

#SBATCH --job-name="ic-array"
#SBATCH --array=1-100
#SBATCH --time=00:60:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=48 --ntasks-per-core=1
#SBATCH --account=uzh29

build_path="pkdgrav3/build/pkdgrav3"
grid_storage="/capstor/scratch/cscs/arehmann/ic_gen/grid/$id"
pts_storage="/capstor/scratch/cscs/arehmann/ic_gen/raw/$id"
seed=$SLURM_ARRAY_TASK_ID

srun pkdgrav3/build/pkdgrav3 params.py $seed $pts_storage $grid_storage
