# Master Thesis

The proper generation of Initial Conditions (ICs) is a crucial part of N-Body simulations. Generally this is achieved by manipulating a random gaussian field, such that it's power spectrum is correct.

We are given an N-Body simulation $f$ which given an initial mass distrubution $x_0$ computes $f(x_0) = x_t$. We are interested in finding $f^{-1}$, however as the function $f$ is highly nonlinear, it is only possible to approximate $f^{-1} \approx f^{\prime}$. If we have found such an $f^{\prime}$ we can generate IC's and potentially also inverse N-Body simulations with adaptive computational cost. 

## Questions

- Is our primary target to find IC's which can be used for N-Body simulations or is it to find the IC's of a given galaxy configuration?
- If we are doing the first, then we are just learning to generate the IC's which are used initially in the N-Body simulation?
- If we are doing the latter, Robert mentioned we want to do it in 3D, however there is no 3D distribution data available?
- What is the problem with current IC generation? What issue are we trying to tackle?

- Or let say we have our given IC generator, do we optimize it for measured quantities of the results of the N-Body simulation? Possible easily differentiable, probably a lot more feasible than diff N-Body simulation. 
- We could use 

## Vocabulary

- **Zoom-In Simulations** a certain section of the simulation domain is simulated at a higher resolution.
- **Dark Matter Simulation** dark matter makes up 85% of the universe. While it is invisible, meaning it does not interact with electromagnetic radiation, the gravitational effects on visible matter are visible. Since such a large percentage of the universe is made of dark matter, and conviniently its much simpler to simulate, precisely because it does not interact with electromagnetic, nuclear and hydrodynamic effects. 
- **Baryonic Simulations** these simulations are a lot more complex and include all the forces which can be simulated and are often done using SPH. 
- **Gigaparsec** A gigaparsec (Gpc) is a unit of length used in astronomy and cosmology. It's equal to one billion parsecs or approximately 3.26 billion light-years.  The size of the observable universe is about 28.5 gigaparsec. 
- **Observable Universe** The part of the universe we can observe, we don't know how large the Universe outside this section is.


## Ressources

### IC

- *Initial Conditions for large Comological Simulations* https://arxiv.org/pdf/0804.3536
- *Transients from Initial Conditions: A Perturbative Analysis* https://arxiv.org/pdf/astro-ph/9711187

### Methodology

- GAN for distribution generation, advantage is that critic is better able ad judging results than straight forward loss functions from final states, only applied for 2D cases here. *From EMBER to FIRE: predicting high resolution baryon fields from
dark matter simulations with Deep Learning* https://arxiv.org/pdf/2110.11970
- Single image generation using denoising diffusion model. https://arxiv.org/pdf/2211.12445


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