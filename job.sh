#!/bin/bash -l

#SBATCH --job-name="icg100"
#SBATCH --time=00:60:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=48 --ntasks-per-core=1
#SBATCH --constraint=gpu
#SBATCH --account=uzh29


srun pkdgrav3/build/pkdgrav3 params.py "/capstor/scratch/cscs/arehmann/ic_gen/raw/100/pts" "/capstor/scratch/cscs/arehmann/ic_gen/grid/100/grid" 
