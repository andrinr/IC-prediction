for i in $(seq 1 1);
do
    # set storage path
    storage_path="/capstor/scratch/cscs/arehmann/ic_gen"
    storage_path_escape=$(echo $storage_path | sed 's/\//\\\//g')
    build_path="pkdgrav3/build/pkdgrav3"
    build_path_escape=$(echo $build_path | sed 's/\//\\\//g')

    # change seed in .par file
    sed -i "s/iSeed.*/iSeed \t\t\t= $i \t\t\t# Seed/" params.py

    # add zero padding
    printf -v j "%03d" $i

    mkdir -p "$storage_path/raw/$j"
    mkdir -p "$storage_path/grid/$j"

    sed -i "s/#SBATCH --job-name=.*/#SBATCH --job-name=\"$j\"/" job.sh

    sed -i "s/srun.*/srun $build_path_escape params.py \"$storage_path_escape\/raw\/$j\/pts\" \"$storage_path_escape\/grid\/$j\/grid\" /" job.sh

    sbatch --wait job.sh

    # # run simulation
    #./pkdgrav3/build/pkdgrav3 params.py "/project/ic_gen/raw/$j/pts" "/project/ic_gen/grid/$j/grid"
done
