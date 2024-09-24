module load cray && module swap PrgEnv-cray PrgEnv-gnu && module load cpeGNU GSL Boost cray-hdf5 cray-fftw CMake cray-python hwloc
sbatch --array=1-300 job.sh