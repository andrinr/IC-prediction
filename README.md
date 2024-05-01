# Master Thesis

The proper generation of Initial Conditions (ICs) is a crucial part of N-Body simulations. 
We are given an N-Body simulation $f$ which given an initial mass distrubution $x_0$ computes $f(x_0) = x_t$. We are interested in finding $f^{-1}$, however as the function $f$ is highly nonlinear, it is only possible to approximate $f^{-1}$. 
The problem is related to inverse problems, which is relevant in engineering disciplines

## Ressources

### IC

- *Initial Conditions for large Comological Simulations* https://arxiv.org/pdf/0804.3536
- *Transients from Initial Conditions: A Perturbative Analysis* https://arxiv.org/pdf/astro-ph/9711187

### Methodology

- GAN for distribution generation, advantage is that critic is better able ad judging results than straight forward loss functions from final states, only applied for 2D cases here. *From EMBER to FIRE: predicting high resolution baryon fields from
dark matter simulations with Deep Learning* https://arxiv.org/pdf/2110.11970


## Getting started

### Installing pkdgrav

Clone this repo then updat the submodules
```git submodule update --init --recursive```
Install all the deps for pkdgrav (Ubuntu only, check here for other https://pkdgrav3.readthedocs.io/en/latest/install.html)

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

### 