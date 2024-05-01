# Master Thesis

The general idea is to create initial conditions 

## Ressources

### Methodology

- From EMBER to FIRE: predicting high resolution baryon fields from
dark matter simulations with Deep Learning https://arxiv.org/pdf/2110.11970


## Getting started

Clone this repo then updat the submodules
```git submodule update --init --recursive```
Install all the deps for PKDGrav (Ubuntu only, check here for other https://pkdgrav3.readthedocs.io/en/latest/install.html)

```{bash}
sudo apt update
sudo apt install -y autoconf automake pkg-config cmake gcc g++ make gfortran git
sudo apt install -y libfftw3-dev libfftw3-mpi-dev libgsl0-dev libboost-all-dev libhdf5-dev libmemkind-dev libhwloc-dev
sudo apt install -y python3-dev cython3 python3-pip python3-numpy python3-ddt python3-nose python3-tomli
```

Create a new python env

```{bash}
python -m venv .env
source .env/
pip install cython numpy ddt nose dill
```

Cd into pkdgrav subrepo and

```
cmake -S . -B build
cmake --build build
```