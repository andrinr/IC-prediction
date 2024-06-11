# set storage path
printf -v id "%03d" $SLURM_ARRAY_TASK_ID

build_path="pkdgrav3/build/pkdgrav3"
grid_storage="/capstor/scratch/cscs/arehmann/ic_gen/grid/$id"
pts_storage="/capstor/scratch/cscs/arehmann/ic_gen/raw/$id"
seed=$SLURM_ARRAY_TASK_ID

sbatch —array=1-2 job.sh $SLURM_ARRAY_TASK_ID $pts_storage $grid_storage