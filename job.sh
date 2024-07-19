#!/bin/bash -l

#SBATCH --job-name="ic-array"
#SBATCH --time=01:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 --cpus-per-task=128 --ntasks-per-core=1
#SBATCH --account=uzh29
#SBATCH --constraint=mc
#SBATCH --partition=normal
#SBATCH --hint=multithread

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

printf -v id "%03d" $SLURM_ARRAY_TASK_ID

build_path="pkdgrav3/build/pkdgrav3"
grid_storage="/capstor/scratch/cscs/arehmann/pairs/grid/$id/grid"
pts_storage="/capstor/scratch/cscs/arehmann/pairs/raw/$id/raw"
seed=$SLURM_ARRAY_TASK_ID

# make directories
mkdir -p $grid_storage
mkdir -p $pts_storage

srun pkdgrav3/build/pkdgrav3 params.py $seed $pts_storage $grid_storage
