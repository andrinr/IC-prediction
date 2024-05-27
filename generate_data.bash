for i in $(seq 1 2);
do
    echo $i
    # change seed in .par file
    sed -i "s/iSeed.*/iSeed \t\t\t= $i \t\t\t# Seed/" params.py

    # add zero padding
    printf -v j "%03d" $i

    mkdir -p "data/raw/$j"
    mkdir -p "data/grid/$j"

    sed -i "s/achOutName=.*/achOutName=\"data\/raw\/$j\" /" params.py
    sed -i "s/gridOutName=.*/gridOutName=\"data\/grid\/$j\" /" params.py
    # # change outname
    # sed -i "s/achOutName = .*/achOutName = data/sq+$i" cosmology_small.par
    
    # # run simulation
    mpirun pkdgrav3/build/pkdgrav3 params.py
done
