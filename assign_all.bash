for input_file in "./data/raw/01/*";
do  
    # echo $filepath
    # get the filename without the path
    file_name=${input_file##*/}

    echo $file_name

    # mpirun pkdgrav3/build/pkdgrav3 assign.py $filepath 128 "./data/grid/01/$filename"
done
