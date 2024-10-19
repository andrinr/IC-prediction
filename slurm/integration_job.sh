#!/bin/bash -l

#SBATCH --job-name="ic-integration"
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 --cpus-per-task=128 --ntasks-per-core=1
#SBATCH --account=uzh29
#SBATCH --constraint=mc
#SBATCH --partition=debug
#SBATCH --hint=multithread

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

build_path="../pkdgrav3/build/pkdgrav3"
grid_storage="/capstor/scratch/cscs/arehmann/integration/grid"
pts_storage="/capstor/scratch/cscs/arehmann/integration/pts"
tipsy_file="../test.tipsy"

# make directories
mkdir -p $grid_storage
mkdir -p $pts_storage

srun $build_path pkd_script_tipsy.py $tipsy_file $pts_storage $grid_storage
