# JAX 
import jax.numpy as jnp
import jax
from jax.lib import xla_bridge
# NVIDIA Dali
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
# Other
import os
import optax
import equinox as eqx
# Local
from VolumetricSequence import VolumetricSequence
import nn

# Parameters
DATA_ROOT = "/shares/feldmann.ics.mnf.uzh/Andrin/IC_GEN/grid/"
INPUT_GRID_SIZE = 128
GRID_SIZE = 32
BATCH_SIZE = 4

# Settings / Device Info
print("Jax backend is using %s" % xla_bridge.get_backend().platform)
jax.config.update("jax_enable_x64", False)
jax.config.update("jax_disable_jit", False)

# Data Pipeline
vol_seq = VolumetricSequence(BATCH_SIZE, INPUT_GRID_SIZE, DATA_ROOT, (0, 30), False)

@pipeline_def(
    batch_size=BATCH_SIZE,
    num_threads=2, 
    device_id=0,
    py_num_workers=4,
    py_start_method="spawn")
def pair_pipeline():

    [start, end] = fn.external_source(
        source=vol_seq,
        num_outputs=2,
        batch=False,
        dtype=types.FLOAT)

    reshape = lambda x : fn.reshape(x, layout="CDHW")
    resize = lambda x : fn.resize(
        x,
        interp_type = types.INTERP_CUBIC,
        antialias=False,
        size=(GRID_SIZE, GRID_SIZE, GRID_SIZE))

    start = resize(reshape(start))
    end = resize(reshape(end))

    return start, end

pipe = pair_pipeline()
pipe.build()
out = pipe.run()

print(jnp.shape(out))

init_rng = jax.random.key(0)
unet = nn.UNet(
    num_spatial_dims=3,
    in_channels=1,
    out_channels=1,
    hidden_channels=8,
    num_levels=3,
    activation=jax.nn.relu,
    padding='SAME',	
    key=init_rng
)
parameter_count = nn.count_parameters(unet)
print(f'Number of parameters: {parameter_count}')

learning_rate = 3e-3

optimizer = optax.adam(learning_rate)
optimzier_state = optimizer.init(eqx.filter(unet, eqx.is_array))

def loss_fn(model : eqx.Module, x : jnp.ndarray, y : jnp.ndarray):
  y_pred = jax.vmap(model)(x)
  mse = jnp.mean(jnp.square(y_pred - y))
  return mse

@eqx.filter_jit
def update_fn(
  model : eqx.Module, 
  optimizer_state : optax.OptState, 
  x : jnp.ndarray, 
  y : jnp.ndarray):

  loss, grad = eqx.filter_value_and_grad(loss_fn)(model, x, y)
  updates, new_optimizer_state = optimizer.update(grad, optimizer_state)
  new_model = eqx.apply_updates(model, updates)
  return new_model, new_optimizer_state, loss

del pipe