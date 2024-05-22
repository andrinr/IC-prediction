directory="./data/raw/01/*"

for file in $directory; do  
    # echo $filepath
    # get the filename without the path
    echo $file

    file_name=${file##*/}

    mpirun pkdgrav3/build/pkdgrav3 assign.py $file 128 "./data/grid/01/$file_name"
done
