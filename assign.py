# simulation mode is skipped when PKDGRAV imported
# mpirun pkdgrav3 this-script.py input.std 100 output
import PKDGRAV as msr
from sys import argv,exit

if len(argv)<=3:
    print(f"Usage: {argv[0]} <filename> <grid> <output>")
    exit(1)
name=argv[1]
nGrid = int(argv[2])
output = argv[3]

# Load the file and setup the tree, then measure the power
time = msr.load(name,nGridPk=nGrid)
msr.domain_decompose()
msr.build_tree()
msr.grid_create(nGrid)
msr.assign_mass()
msr.grid_write(output)
