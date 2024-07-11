for i in $(seq 1 1);
do
    # set storage path
    storage_path="/capstor/scratch/cscs/arehmann/ic_gen"
    storage_path_escape=$(echo $storage_path | sed 's/\//\\\//g')
    build_path="pkdgrav3/build/pkdgrav3"
    build_path_escape=$(echo $build_path | sed 's/\//\\\//g')

    ./pkdgrav3/build/pkdgrav3 params.py $i "pts" "grid"

    scp grid arehma@cluster.s3it.uzh.ch:/data/arehma/grid

    rm grid
done
