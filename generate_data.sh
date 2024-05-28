for i in $(seq 1 2);
do
    echo $i
    # change seed in .par file
    sed -i "s/iSeed.*/iSeed \t\t\t= $i \t\t\t# Seed/" job_params.py

    # add zero padding
    printf -v j "%03d" $i

    mkdir -p "data/raw/$j"
    mkdir -p "data/grid/$j"

    sed -i "s/achOutName=.*/achOutName=\"data\/raw\/$j\" /" job_params.py
    sed -i "s/gridOutName=.*/gridOutName=\"data\/grid\/$j\" /" job_params.py

    sed -i  "s/job-name=.*$/job-name=\"gen-$j\" /g" job.sh;

    # # run simulation
    mpirun pkdgrav3/build/pkdgrav3 params.py
done
