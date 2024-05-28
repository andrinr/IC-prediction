for i in $(seq 1 1);
do
    echo $i
    # change seed in .par file
    sed -i "s/iSeed.*/iSeed \t\t\t= $i \t\t\t# Seed/" params.py

    # add zero padding
    printf -v j "%03d" $i

    mkdir -p "data/raw/$j"
    mkdir -p "data/grid/$j"

    sed -i "s/achOutName=.*/achOutName=\"data\/raw\/$j\/pts" /" job_params.py
    sed -i "s/gridOutName=.*/gridOutName=\"data\/grid\/$j\/grid" /" job_params.py

    # # run simulation
   ./pkdgrav3/build/pkdgrav3 params.py
done
