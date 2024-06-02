#!/bin/bash -l

#SBATCH --job-name="001"
#SBATCH --time=00:10:00
#SBATCH --nodes=4
#SBATCH --ntasks=48
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=16 
#SBATCH --ntasks-per-core=2
#SBATCH --constraint=mc
#SBATCH --account=uzh29

srun pkdgrav3/build/pkdgrav3 params.py "/capstor/scratch/cscs/arehmann/ic_gen/raw/001/pts" "/capstor/scratch/cscs/arehmann/ic_gen/grid/001/grid" 
