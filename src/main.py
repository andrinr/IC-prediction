# JAX 
import jax.numpy as jnp
import jax
from jax.lib import xla_bridge
# NVIDIA Dali
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.jax import DALIGenericIterator
# Other
import matplotlib.pyplot as plt
# Local
from data.VolumetricSequence import VolumetricSequence
import nn
import data

# Parameters
DATA_ROOT = "/shares/feldmann.ics.mnf.uzh/Andrin/IC_GEN/grid/"
INPUT_GRID_SIZE = 128
GRID_SIZE = 32
BATCH_SIZE = 8

# Settings / Device Info
print("Jax backend is using %s" % xla_bridge.get_backend().platform)
jax.config.update("jax_enable_x64", False)
jax.config.update("jax_disable_jit", False)

# Data Pipeline
vol_seq = data.VolumetricSequence(
    BATCH_SIZE, INPUT_GRID_SIZE, DATA_ROOT, (0, 30), False)

@pipeline_def(
    batch_size=BATCH_SIZE,
    num_threads=2, 
    device_id=0,
    py_num_workers=16,
    py_start_method="spawn")
def volumetric_pairs_pipe(external_iterator):

    [start, end] = fn.external_source(
        source=external_iterator,
        num_outputs=2,
        batch=False,
        dtype=types.FLOAT)

    reshape_fn = lambda x : fn.reshape(x, layout="CDHW")
    resize_fn = lambda x : fn.resize(
        x,
        interp_type = types.INTERP_CUBIC,
        antialias=False,
        size=(GRID_SIZE, GRID_SIZE, GRID_SIZE))

    start = resize_fn(reshape_fn(start))
    end = resize_fn(reshape_fn(end))
    
    return start, end

data_pipeline = volumetric_pairs_pipe(vol_seq)
data_iterator = DALIGenericIterator(data_pipeline, ["start", "end"])

# Initialize Neural Network
init_rng = jax.random.key(0)
unet = nn.UNet(
    num_spatial_dims=3,
    in_channels=1,
    out_channels=1,
    hidden_channels=8,
    num_levels=3,
    activation=jax.nn.relu,
    padding='SAME',	
    key=init_rng)

parameter_count = nn.count_parameters(unet)
print(f'Number of parameters: {parameter_count}')

learning_rate = 3e-5

nn.train(
    unet,
    data_iterator,
    learning_rate,
    10)

# data = next(data_iterator)
fig, axs = plt.subplots(3, 1)
# axs[0].imshow(

# )

# Delete Data Pipeline
del data_pipeline