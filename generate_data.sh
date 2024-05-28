for i in $(seq 1 1);
do
    echo $i
    # change seed in .par file
    sed -i "s/iSeed.*/iSeed \t\t\t= $i \t\t\t# Seed/" params.py

    # add zero padding
    printf -v j "%03d" $i

    mkdir -p "data/raw/$j"
    mkdir -p "data/grid/$j"

    # # run simulation
   ./pkdgrav3/build/pkdgrav3 params.py "data/raw/$j/pts" "data/grid/$j/grid"
done
