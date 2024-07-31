# JAX 
import jax.numpy as jnp
import jax
from jax.lib import xla_bridge
import nvidia.dali.types as types
from nvidia.dali.plugin.jax import DALIGenericIterator
# Local
import nn
import data

# Parameters
DATA_DIR = "/shares/feldmann.ics.mnf.uzh/Andrin/IC_GEN/grid/"
MODEL_OUT_DIR = "ICG/model_params/"
INPUT_GRID_SIZE = 128
GRID_SIZE = 64
LEARNING_RATE = 0.001
N_EPOCHS = 100

# JAX Settings / Device Info
print("Jax backend is using %s" % xla_bridge.get_backend().platform)
jax.config.update("jax_enable_x64", False)
jax.config.update("jax_disable_jit", False)

# Data Pipeline
dataset = data.VolumetricSequence(
    grid_size = INPUT_GRID_SIZE,
    directory = DATA_DIR,
    start = 0,
    steps = 50,
    stride = 10)

sample_info = types.SampleInfo(
    idx_in_epoch=5,
    idx_in_batch=0,
    iteration=0,
    epoch_idx=0)

[sequence, steps] = dataset(sample_info)
print(sequence.shape)
print(steps)

data_pipeline = data.volumetric_sequence_pipe(dataset, GRID_SIZE)
data_iterator = DALIGenericIterator(data_pipeline, ["sequence", "steps"])

# Initialize Neural Network
init_rng = jax.random.key(0)
# unet_hyperparams = {
#     "num_spatial_dims" : 3,
#     "in_channels" : 1,
#     "out_channels" : 1,
#     "hidden_channels" : 8,
#     "num_levels" : 4,
#     "padding" : 'SAME',
#     "padding_mode" : 'CIRCULAR'}

# model = nn.UNet(
#     activation=jax.nn.relu,
#     **unet_hyperparams,	
#     key=init_rng)

fno_hyperparams = {
    "modes" : 8,
    "hidden_channels" : 4,
    "n_furier_layers" : 4}

model = nn.fno.FNO(
    activation = jax.nn.relu,
    key = init_rng,
    **fno_hyperparams)

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
model, losses = nn.train_model(
    model,
    data_iterator,
    LEARNING_RATE,
    N_EPOCHS,
    nn.mse_loss)

nn.save(MODEL_OUT_DIR, "fno.eqx", fno_hyperparams, model)

# Delete Data Pipeline
del data_pipeline