BATCH --job-name="hello_world_mpi"
#SBATCH --time=00:00:30
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=1 --cpus-per-task=16 --ntasks-per-core=1
#SBATCH --constraint=gpu
#SBATCH --output=
#SBATCH --error=
#SBATCH --account=uzh29

srun ./pkdgrav3/build/pkdgrav3 params.py
