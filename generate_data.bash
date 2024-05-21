mkdir -p data

for i in $(seq 1 2);
do
    # change seed in .par file
    sed -i "s/iSeed = [0-9]*/iSeed = $i/" cosmology_small.par
    # # change outname
    # sed -i "s/achOutName = .*/achOutName = data/sq+$i" cosmology_small.par
    
    # # run simulation
    # pkdgrav/pkdgrav cosmology_small.par
done
