# Master Thesis

We want to find the initial conditions (IC's), that is the mass density distribution, for small scale dark matter simulations. That is, given simulation $f$ which given  $x_0$ computes $f(x_0) = x_t$. We are interested in finding $f^{-1}$, however as the function $f$ is highly nonlinear, it is only possible to approximate $f^{-1} \approx f^{\prime}$. 

The goal is to apply this technique to observed density distributions and see if we can actually find the correct IC's. Naturally this would be interesting as the evolution of structures could be analyzed.

In nature $x_0$ is given by the spatial distribution of mass and to some extent their velocity. The velocity on the axis tangential to the viewer cannot be determined. The primary inspiration for the approximate inverse simulation model $f^{\prime}$ comes from conditional generative models for image generation / analysis. Therefore we perform a 3D mass assignment of the given objects, where discrete convolution, pooling and similar operations can be learned on.

There are certain properties, which are assumed to hold for correct IC's. Generally we think the mass distribution is a Gaussian Noise. Usually the IC is constructed by sampling random white noise on a grid with fixed distances. Then a specific discrete convolution operator is applied to obtain a gaussian distribution. This process is oftentimes repeated with a more fine grained grid overlapping the corase first grid. Furthermore the power spectrum of the distribution should follow a certain distribution. As for the training of the model, we can experiment with encoding the IC constraints in the loss function, however this is optional. The most simple case is to simply leverage a voxel wise MSE error, Stadel has proposed to use the cross power spectrum between the truth and predictions.

## Intermediate Learning Projects

To learn some techniques I have made some smaller projects:

- https://github.com/andrinr/jax-gan - GAN for 2D distribution generation
- https://github.com/andrinr/jax-image-autoencoder - Image autoencoder for compression / image generation
- https://github.com/andrinr/jax-neural-ode - Neural ODE implementation in JAX including adjoint method for backpropagation
- https://github.com/andrinr/motion-derivative - Inverse simulation for small N-Body gravitation simulations

## Things to think about

- How do we quantify the error of the model? Does MSE really capture the interesting aspects?
- RL approach could be potentially viable, as shown here: https://physicsbaseddeeplearning.org/reinflearn-code.html however we would need an interactive environment to train the model on. This could potentially also be the endgoal, getting a N-Body RL-GYM env?
- Otherwise we can try two things, we try to find some correlations between initial and final states. This could be trained on a massive amount of data, however quiet unrealistic that there parallels can really be found.
- We limit the project on trying to invert simulations using DL techniques. I am pretty sure we cannot invert large redshits.
- We can also try to just build a forward surrogate model, which then is differentiable, meaning we could apply an optimziation strategy for the IC's. 

## Learnings / Unclarities

- Data has distribution which is difficult to work with, as vast very empty regions and small dense regions -> log normalization and Z-Score normalization seems to be the way to go.
- The way we measure the error is important, MSE does not appear to work great. 
- Training the model on predicting the difference between timesteps seems to work better. 
- U-NET works, however cannot efficently train it on my local machine on res greater than 32^3


- Train UNET on previous time prediction
    - W

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
- Neural PDE as described here https://arxiv.org/pdf/2110.10249

- They do kindof what we do here: https://astro.ft.uam.es/gustavo/

### Cosmology

- Pretty much everything important here: https://ned.ipac.caltech.edu/level5/Sept02/Kinney/Kinney_contents.html

## Getting started with data gneration

### Installing pkdgrav

Clone this repo then updat the submodules
```git submodule update --init --recursive```
and change to the dev branch of pkdgrav
```git checkout develop```. If you want to update the submodule to the latest version do
``` git submodule update --remote```.
Install all the deps for pkdgrav (Ubuntu only, check here for other https://pkdgrav3.readthedocs.io/en/latest/install.html)

```{bash}
sudo apt update
sudo apt install -y autoconf automake pkg-config cmake gcc g++ make gfortran git
sudo apt install -y libfftw3-dev libfftw3-mpi-dev libgsl0-dev libboost-all-dev libhdf5-dev libmemkind-dev libhwloc-dev
sudo apt install -y python3-dev cython3 python3-pip python3-numpy python3-ddt python3-nose python3-tomli
```
Optionally install CUDA on the machine, make sure to link it in PATH and LD_LIBRARY_PATH.

Create a new python env

```{bash}
python -m venv .env
source .env/bin/activate
pip install -r pkdgrav3/requirements.txt
```

cd into pkdgrav submodule and

```
cmake -S . -B build
cmake --build build
```

### Connect to cscs on windwos

Setup:
1. follow install instruction for script here: https://user.cscs.ch/access/auth/mfa/
2. edit config  ```notepad.exe C:\Users\<USER>\.ssh\config``` and add: 
```
Host cscs
	User <USERNAME>
	HostName daint.cscs.ch
	ProxyJump <USERNAME>@ela.cscs.ch
```

This steps have to be repeated every day to get a new key.
1. check if ssh-agent is running with ```Get-Service ssh-agent```
2. if not running start it with ```Get-Service ssh-agent | Set-Service -StartupType Automatic``` and ```Start-Service ssh-agent```
3. Activate the python env ```.\venv\Scripts\activate```
4. execute the ssh-gen script ```python .\sshservice-cli\cscs-keygen.py``` and enter credentials. No password needed. 
5. add key ```ssh-add ~\.ssh\cscs-key```
6. connect with ```ssh cscs``` or use vs code remote ssh extension

### Run data generation on eiger

1. load modules ```module load cray && module swap PrgEnv-cray PrgEnv-gnu && module load cpeGNU GSL Boost cray-hdf5 cray-fftw CMake cray-python hwloc```
2. run ```bash generate_data.sh```
3. check task by squeue and filter by username ```squeue -u <USERNAME>```
4. if needed cancel task with ```scancel <JOBID>```

## Getting started with data analysis, generative models

Create a new python env

```{bash}
python -m venv jax.env
source .env/bin/activate
pip install -U "jax[cuda12]"
pip install pytorch torchvision torchaudio cpuonly -c pytorch
pip install optax equinox matplotlib pyccl
```

PYCCL can be a bit tricky to get working. In my case I needed to install ```pip install wheel pyyaml```,
```sudo apt install libpcre3 libpcre3-dev``` and ```sudo apt install swig``` prior to installing pyccl. 
Furthermore we need to install classy as described in their instructions here: https://github.com/lesgourg/class_public/wiki/Python-wrapper.

## Science Cluster

### Installation

1. Create a new interactive session ```srun --pty -n 1 -c 8 --time=01:00:00 --mem=16G --gres=gpu:1 bash -l``` 
2. Load Conda ```module load anaconda3```
3. Create env ```{bash}
python -m venv ml-env
source .env/bin/activate
pip install -U "jax[cuda12]"
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install nvidia-dali-cuda120
pip install optax equinox matplotlib
```

### Startup

1. Create a new interactive session ```srun --pty -n 1 -c 8 --time=01:00:00 --mem=16G --gres=gpu:1 bash -l``` 
2. Load Conda ```module load anaconda3```
3. Load env ```source activate myenv```