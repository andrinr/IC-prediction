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
import equinox as eqx
# Local
import nn
import data
import cosmos
import field

# Parameters
DATA_ROOT = "/shares/feldmann.ics.mnf.uzh/Andrin/IC_GEN/grid/"
INPUT_GRID_SIZE = 128
GRID_SIZE = 32
BATCH_SIZE = 8
LEARNING_RATE = 0.01
N_EPOCHS = 20

# JAX Settings / Device Info
print("Jax backend is using %s" % xla_bridge.get_backend().platform)
jax.config.update("jax_enable_x64", False)
jax.config.update("jax_disable_jit", False)

# Data Pipeline
# dataset = data.VolumetricSequence(
#     BATCH_SIZE, INPUT_GRID_SIZE, DATA_ROOT, (7, 10), False)

dataset = data.TestData(BATCH_SIZE, GRID_SIZE)

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

data_pipeline = volumetric_pairs_pipe(dataset)
data_iterator = DALIGenericIterator(data_pipeline, ["start", "end"])

# Initialize Neural Network
init_rng = jax.random.key(0)
# model = nn.UNet(
#     num_spatial_dims=3,
#     in_channels=1,
#     out_channels=1,
#     hidden_channels=8,
#     num_levels=2,
#     activation=jax.nn.relu,
#     padding='SAME',
#     padding_mode='CIRCULAR',	
#     key=init_rng)

model = nn.fno.FNO(
    modes = 8,
    hidden_channels = 4,
    activation = jax.nn.relu,
    n_furier_layers = 2,
    key = init_rng)

# model = nn.Dummy(
#     num_spatial_dims=3,
#     channels=1,
#     activation=jax.nn.relu,
#     padding='SAME',
#     padding_mode='CIRCULAR',
#     key=init_rng)

parameter_count = nn.count_parameters(model)
print(f'Number of parameters: {parameter_count}')

# train the model
model, losses = nn.train(
    model,
    data_iterator,
    LEARNING_RATE,
    N_EPOCHS,
    nn.mse_loss)

data_pipeline = volumetric_pairs_pipe(dataset)
data_iterator = DALIGenericIterator(data_pipeline, ["start", "end"])

data = next(data_iterator)
x = jax.device_put(data['end'], jax.devices('gpu')[0])
y_star = jax.device_put(data['start'], jax.devices('gpu')[0])

params, static = eqx.partition(model, eqx.is_array)
print(nn.mse_loss(params, static, x, y_star))

y = model(x[0])

y_star = jnp.reshape(y_star[0], (GRID_SIZE, GRID_SIZE, GRID_SIZE, 1))
y = jnp.reshape(y, (GRID_SIZE, GRID_SIZE, GRID_SIZE, 1))
x = jnp.reshape(x[0], (GRID_SIZE, GRID_SIZE, GRID_SIZE, 1))

print(y_star.shape)
print(y.shape)
print(x.shape)

print(jnp.max(y_star))
print(jnp.max(y))
print(jnp.min(y_star))
print(jnp.min(y))

min = jnp.min(jnp.array([y_star, y, x]))
max = jnp.max(jnp.array([y_star, y, x]))

power_spectrum = cosmos.PowerSpectrum(GRID_SIZE, 40)

fig, axs = plt.subplots(2, 4, figsize=(10, 8))

# set inferno colormap
axs[0, 0].axis('off')
axs[0, 0].set_title("Y star")
axs[0, 0].imshow(y_star[GRID_SIZE // 2, : , :], vmin=min, vmax=max, cmap='inferno')
k, power = power_spectrum(y_star[:, :, :, 0])
axs[1, 0].plot(k, power)

axs[0, 1].axis('off')
axs[0, 1].set_title("Y")
axs[0, 1].imshow(y[GRID_SIZE // 2, : , :], vmin=min, vmax=max, cmap='inferno')
k, power = power_spectrum(y[:, :, :, 0])
axs[1, 1].plot(k, power)

axs[0, 2].axis('off')
axs[0, 2].set_title("X")
axs[0, 2].imshow(x[GRID_SIZE // 2, : , :], vmin=min, vmax=max, cmap='inferno')
k, power = power_spectrum(x[:, :, :, 0])
axs[1, 2].plot(k, power)

difference = y[GRID_SIZE // 2, : , :] - y_star[GRID_SIZE // 2, : , :]
axs[0, 3].axis('off')
axs[0, 3].set_title("X")
axs[0, 3].imshow(difference)

plt.savefig("pred_vs_real.jpg")

# Delete Data Pipeline
del data_pipeline