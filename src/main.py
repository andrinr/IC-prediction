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
import nn
import data
import cosmos
import field

# Parameters
DATA_ROOT = "/shares/feldmann.ics.mnf.uzh/Andrin/IC_GEN/grid/"
INPUT_GRID_SIZE = 128
GRID_SIZE = 64
BATCH_SIZE = 8
LEARNING_RATE = 0.001
N_EPOCHS = 30

# JAX Settings / Device Info
print("Jax backend is using %s" % xla_bridge.get_backend().platform)
jax.config.update("jax_enable_x64", False)
jax.config.update("jax_disable_jit", False)

# Data Pipeline
vol_seq = data.VolumetricSequence(
    BATCH_SIZE, INPUT_GRID_SIZE, DATA_ROOT, (45, 46), False)

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
# model = nn.UNet(
#     num_spatial_dims=3,
#     in_channels=1,
#     out_channels=1,
#     hidden_channels=8,
#     num_levels=4,
#     activation=jax.nn.relu,
#     padding='SAME',
#     padding_mode='CIRCULAR',	
#     key=init_rng)

model = nn.fno.FNO(
    modes = 4,
    hidden_channels = 16,
    activation = jax.nn.relu,
    n_furier_layers = 4,
    key = init_rng)

parameter_count = nn.count_parameters(model)
print(f'Number of parameters: {parameter_count}')

# train the model
model, losses = nn.train(
    model,
    data_iterator,
    LEARNING_RATE,
    N_EPOCHS,
    nn.mse_loss)

data_pipeline = volumetric_pairs_pipe(vol_seq)
data_iterator = DALIGenericIterator(data_pipeline, ["start", "end"])

# with plt.style.context('science'):
data = next(data_iterator)
end_d = jax.device_put(data['end'], jax.devices('gpu')[0])
start_d = jax.device_put(data['start'], jax.devices('gpu')[0])

start_prediction = model(end_d[3])

start_d = jnp.reshape(start_d[3], (GRID_SIZE, GRID_SIZE, GRID_SIZE, 1))
start_prediction = jnp.reshape(start_prediction, (GRID_SIZE, GRID_SIZE, GRID_SIZE, 1))
end_d = jnp.reshape(end_d[3], (GRID_SIZE, GRID_SIZE, GRID_SIZE, 1))

power_spectrum = cosmos.PowerSpectrum(
    GRID_SIZE, 40)

fig, axs = plt.subplots(2, 4)

axs[0, 0].set_title("Y star")
axs[0, 0].imshow(
    start_d[GRID_SIZE // 2, : , :])
k, power = power_spectrum(start_d[:, :, :, 0])
axs[1, 0].plot(k, power)

axs[0, 1].set_title("Y")
axs[0, 1].imshow(
    start_prediction[GRID_SIZE // 2, : , :])
k, power = power_spectrum(start_prediction[:, :, :, 0])
axs[1, 1].plot(k, power)

axs[0, 2].set_title("X")
axs[0, 2].imshow(
    end_d[GRID_SIZE // 2, : , :])
k, power = power_spectrum(end_d[:, :, :, 0])
axs[1, 2].plot(k, power)

difference = start_prediction[GRID_SIZE // 2, : , :] - start_d[GRID_SIZE // 2, : , :]

axs[0, 3].set_title("X")
axs[0, 3].imshow(
    difference)

plt.savefig("pred_vs_real.jpg")

# Delete Data Pipeline
del data_pipeline