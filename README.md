# Master Thesis

We want to find the initial conditions (IC's), that is the mass density distribution, for small scale dark matter simulations. That is, given simulation $f$ which given  $x_0$ computes $f(x_0) = x_t$. We are interested in finding $f^{-1}$, however as the function $f$ is highly nonlinear, it is only possible to approximate $f^{-1} \approx f^{\prime}$. 

The goal is to apply this technique to observed density distributions and see if we can actually find the correct IC's. Naturally this would be interesting as the evolution of structures could be analyzed.

In nature $x_0$ is given by the spatial distribution of mass and to some extent their velocity. The velocity on the axis tangential to the viewer cannot be determined. The primary inspiration for the approximate inverse simulation model $f^{\prime}$ comes from conditional generative models for image generation / analysis. Therefore we perform a 3D mass assignment of the given objects, where discrete convolution, pooling and similar operations can be learned on.

There are certain properties, which are assumed to hold for correct IC's. Generally we think the mass distribution is a Gaussian Noise. Usually the IC is constructed by sampling random white noise on a grid with fixed distances. Then a specific discrete convolution operator is applied to obtain a gaussian distribution. This process is oftentimes repeated with a more fine grained grid overlapping the corase first grid. Furthermore the power spectrum of the distribution should follow a certain distribution. As for the training of the model, we can experiment with encoding the IC constraints in the loss function, however this is optional. The most simple case is to simply leverage a voxel wise MSE error, Stadel has proposed to use the cross power spectrum between the truth and predictions.

## Questions

- ...

## Potential Applications

- Evolution of structures in the universe, matter distribution today is known
- Local universe stuff / or more global. 
- Local: more or less pos and vel. 
- Large scale: Euclid arena, only 3D. 

## Vocabulary

- **Zoom-In Simulations** a certain section of the simulation domain is simulated at a higher resolution.
- **Dark Matter Simulation** dark matter makes up 85% of the universe. While it is invisible, meaning it does not interact with electromagnetic radiation, the gravitational effects on visible matter are visible. Since such a large percentage of the universe is made of dark matter, and conviniently its much simpler to simulate, precisely because it does not interact with electromagnetic, nuclear and hydrodynamic effects. 
- **Baryonic Simulations** these simulations are a lot more complex and include all the forces which can be simulated and are often done using SPH. 
- **Gigaparsec** A gigaparsec (Gpc) is a unit of length used in astronomy and cosmology. It's equal to one billion parsecs or approximately 3.26 billion light-years.  The size of the observable universe is about 28.5 gigaparsec. 
- **Observable Universe** The part of the universe we can observe, we don't know how large the Universe outside this section is.
- **Halos** refer to the regions of space where matter (both normal matter and dark matter) has collapsed under gravity to form structures, such as galaxies or clusters of galaxies.   
- **Phase Based Density**

## Ressources

Grafic 1

### IC

- *Initial Conditions for large Comological Simulations* https://arxiv.org/pdf/0804.3536
- Han & Abel
- *Transients from Initial Conditions: A Perturbative Analysis* https://arxiv.org/pdf/astro-ph/9711187
- https://ui.adsabs.harvard.edu/abs/2011ascl.soft06008B/abstract

### Methodology

- GAN for distribution generation, advantage is that critic is better able ad judging results than straight forward loss functions from final states, only applied for 2D cases here. *From EMBER to FIRE: predicting high resolution baryon fields from
dark matter simulations with Deep Learning* https://arxiv.org/pdf/2110.11970
- Single image generation using denoising diffusion model. https://arxiv.org/pdf/2211.12445
- Diffusion models tutorial: https://cvpr2022-tutorial-diffusion-models.github.io/

- They do kindof what we do here: https://astro.ft.uam.es/gustavo/

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
source .env/bin/activate
pip install cython numpy ddt nose dill
```

Cd into pkdgrav subrepo and

```
cmake -S . -B build
cmake --build build
```

### 