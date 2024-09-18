# Master Thesis

We want to find the initial conditions (IC's), that is the mass density distribution, for small scale dark matter simulations. That is, given simulation $f$ which given  $x_0$ computes $f(x_0) = x_t$. We are interested in finding $f^{-1}$, however as the function $f$ is highly nonlinear, it is only possible to approximate $f^{-1} \approx f^{\prime}$. 

The goal is to apply this technique to observed density distributions and see if we can actually find the correct IC's. Naturally this would be interesting as the evolution of structures could be analyzed.

In nature $x_0$ is given by the spatial distribution of mass and to some extent their velocity. The velocity on the axis tangential to the viewer cannot be determined. The primary inspiration for the approximate inverse simulation model $f^{\prime}$ comes from conditional generative models for image generation / analysis. Therefore we perform a 3D mass assignment of the given objects, where discrete convolution, pooling and similar operations can be learned on.

There are certain properties, which are assumed to hold for correct IC's. Generally we assume the mass distribution is a Gaussian Noise with a specific power spectrum, which is known from the cosmic background radiation. For simulations, the $x_0$ is constructed by sampling random white noise on a grid with fixed distances. Then a convolution is used to imprint the specific power spectrum into the noise. This is a trivial element wise matrix operation, when done in furier space.

A specific example on how to generate random intial conditions can be found in src/experiments/IC.ipynb.

## Structure of the code

The main code is organized in the ``src`` folder we ``train.py`` serves as the entry point for training the model. There are two models implemented, a 3D UNET and a 3D FNO (Furier Neural Operator). The models can be found in ``src/nn/``. The code also includes different methods mass assignment and grid interpolations, which are implemented in JAX and therefore fully differentiable. The differentiable property is leveraged in ``src/field/fit_field.py``, which can be used to fit the particle positions to a 3D density field. 

## Intermediate Learning Projects

To learn some techniques I have made some smaller projects:

- https://github.com/andrinr/jax-gan - GAN for 2D distribution generation
- https://github.com/andrinr/jax-image-autoencoder - Image autoencoder for compression / image generation
- https://github.com/andrinr/jax-neural-ode - Neural ODE implementation in JAX including adjoint method for backpropagation
- https://github.com/andrinr/motion-derivative - Inverse simulation for small N-Body gravitation simulations

## Vocabulary

- **Zoom-In Simulations** a certain section of the simulation domain is simulated at a higher resolution.
- **Dark Matter Simulation** dark matter makes up 85% of the universe. While it is invisible, meaning it does not interact with electromagnetic radiation, the gravitational effects on visible matter are visible. Since such a large percentage of the universe is made of dark matter, and conviniently its much simpler to simulate, precisely because it does not interact with electromagnetic, nuclear and hydrodynamic effects. 
- **Baryonic Simulations** these simulations are a lot more complex and include all the forces which can be simulated and are often done using SPH. 
- **Gigaparsec** A gigaparsec (Gpc) is a unit of length used in astronomy and cosmology. It's equal to one billion parsecs or approximately 3.26 billion light-years.  The size of the observable universe is about 28.5 gigaparsec. 
- **Observable Universe** The part of the universe we can observe, we don't know how large the Universe outside this section is.
- **Halos** refer to the regions of space where matter (both normal matter and dark matter) has collapsed under gravity to form structures, such as galaxies or clusters of galaxies.   

## Data Analsysis on slurm machine

### Installation

1. Create a new interactive session ```srun --pty -n 1 -c 8 --time=01:00:00 --mem=16G --gres=gpu:1 bash -l``` 
2. Load Conda ```module load anaconda3```
3. Create env 
```
python -m venv ml-env
source .env/bin/activate
pip install -U "jax[cuda12]"
pip install nvidia-dali-cuda120
pip install optax equinox matplotlib
conda install -c bccp nbodykit
conda update -c bccp --all
```

### Train & Eval

1. Create a new interactive session ```srun --pty -n 1 -c 8 --time=01:00:00 --mem=16G --gres=gpu:1 bash -l``` 
2. Load Conda ```module load anaconda3```
3. Load env ```source activate myenv```
4. Train model ```python src/train.py default_config.yaml```
5. Evaluate model ```python src/evaluate.py default_config.yaml```

## Data Generation

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
1. Install python
2. ```git clone https://github.com/eth-cscs/sshservice-cli```
3. ```python -m venv mfa```
4. With admin rights ```Set-ExecutionPolicy -ExecutionPolicy Unrestricted```
5. Activate the python env ```.\venv\Scripts\activate```
6. ```pip install -r requirements.txt```
7. edit config  ```notepad.exe C:\Users\<USER>\.ssh\config``` and add: 
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

### Run data generation on SLURM system

1. load modules (Specific for Eiger CSCS) ```module load cray && module swap PrgEnv-cray PrgEnv-gnu && module load cpeGNU GSL Boost cray-hdf5 cray-fftw CMake cray-python hwloc```
2. run ```bash slurm/generate_data.sh```
3. check task by squeue and filter by username ```squeue -u <USERNAME>```
4. if needed cancel task with ```scancel <JOBID>```

## Testing

Make sure pytest is installed and run ```python -m pytest``` from the root directory. 

## Sources


### IC Generation

- *Initial Conditions for large Comological Simulations* https://arxiv.org/pdf/0804.3536
- *Transients from Initial Conditions: A Perturbative Analysis* https://arxiv.org/pdf/astro-ph/9711187
- https://ui.adsabs.harvard.edu/abs/2011ascl.soft06008B/abstract

### Methodology

- FNO + RNN https://arxiv.org/pdf/2303.02243

### Cosmology

- Pretty much everything important here: https://ned.ipac.caltech.edu/level5/Sept02/Kinney/Kinney_contents.html
- How to get velocities from density fields: https://arxiv.org/pdf/astro-ph/9506070
- zeldovhich: https://iopscience.iop.org/article/10.1086/498496/pdf
