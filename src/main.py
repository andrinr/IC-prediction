# JAX 
import jax.numpy as jnp
import jax
from jax.lib import xla_bridge
# NVIDIA Dali
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.jax import DALIGenericIterator
from nvidia.dali.plugin import jax as dax
# Other
import os
import optax
import equinox as eqx
from functools import partial
import statistics
# Local
from VolumetricSequence import VolumetricSequence
import nn

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
vol_seq = VolumetricSequence(
    BATCH_SIZE, INPUT_GRID_SIZE, DATA_ROOT, (0, 30), False)

@pipeline_def(
    batch_size=BATCH_SIZE,
    num_threads=2, 
    device_id=0,
    py_num_workers=16,
    py_start_method="spawn")
def pair_pipeline(external_iterator):

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

data_pipeline = pair_pipeline(vol_seq)
data_iterator = DALIGenericIterator(data_pipeline, ["start", "end"])

# Initialize Neural Network
init_rng = jax.random.key(0)
unet = nn.UNet(
    num_spatial_dims=3,
    in_channels=1,
    out_channels=1,
    hidden_channels=4,
    num_levels=2,
    activation=jax.nn.relu,
    padding='SAME',	
    key=init_rng)

parameter_count = nn.count_parameters(unet)
print(f'Number of parameters: {parameter_count}')

learning_rate = 3e-3

optimizer = optax.adam(learning_rate)
optimzier_state = optimizer.init(eqx.filter(unet, eqx.is_array))

def overdensity(density: jax.Array):
    mean = density.mean()
    return (density - mean) / mean

@partial(jax.jit, static_argnums=1)
def loss_fn(params, static, x : jnp.ndarray, y : jnp.ndarray):
    model = eqx.combine(params, static)
    y_pred = jax.vmap(model)(x)
    mse = jnp.mean(jnp.square(y_pred - y))
    return mse

@eqx.filter_jit
def update_fn(
        start : jnp.ndarray,
        end : jnp.ndarray,
        model : eqx.Module, 
        optimizer_state : optax.OptState):

    params, static = eqx.partition(model, eqx.is_array)

    # this could probably be done inside the dali pipile using mean reduction and normalize
    start_overdensity = jax.vmap(overdensity)(start)
    end_overdensity = jax.vmap(overdensity)(end)

    loss, grad = eqx.filter_value_and_grad(loss_fn)(
        params, static, end_overdensity, start_overdensity)
    updates, new_optimizer_state = optimizer.update(grad, optimizer_state)
    new_model = eqx.apply_updates(model, updates)

    return new_model, new_optimizer_state, loss

for epoch in range(40):
    losses = []
    for i, data in enumerate(data_iterator):
        start_d = jax.device_put(data['start'], jax.devices('gpu')[0])
        end_d = jax.device_put(data['end'], jax.devices('gpu')[0])

        # print(jnp.shape(start_d))
        unet, optimizer_state, loss = update_fn(
            start_d,
            end_d,
            unet,  
            optimzier_state)
        
        losses.append(loss)
    
    losses = jnp.array(losses)
    print(losses.mean())

# Delete Data Pipeline
del data_pipeline