# JAX 
import jax
from jax.lib import xla_bridge
from nvidia.dali.plugin.jax import DALIGenericIterator
# equinox
import equinox as eqx
# Local
import nn
import data

# Parameters
DATA_DIR = "/shares/feldmann.ics.mnf.uzh/Andrin/IC_GEN/grid/"
MODEL_OUT_DIR = "ICG/model_params/"
INPUT_GRID_SIZE = 128
GRID_SIZE = 64
LEARNING_RATE = 0.003
N_EPOCHS = 20

# JAX Settings / Device Info
print("Jax backend is using %s" % xla_bridge.get_backend().platform)
jax.config.update("jax_enable_x64", False)
jax.config.update("jax_disable_jit", False)

steps = 50
stride = 10
dataset = data.VolumetricSequence(
    grid_size = INPUT_GRID_SIZE,
    directory = DATA_DIR,
    start = 0,
    steps = 50,
    stride = 10,
    flip=True)

data_pipeline = data.volumetric_sequence_pipe(dataset, GRID_SIZE)
data_iterator = DALIGenericIterator(data_pipeline, ["sequence", "time"])

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

sq_fno_hyperparams = {
    "modes" : 32,
    "hidden_channels" : 4,
    "n_furier_layers" : 4}

model_params = []
model_static = None
for i in range(steps // stride):    
    model = nn.FNO(
        activation = jax.nn.relu,
        key = init_rng,
        **sq_fno_hyperparams)
    
    params, model_static = eqx.partition(model, eqx.is_array)
    print(params)
    model_params.append(params)

if steps // stride == 1:
    model_params = model_params[0]

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
    model_static, 
    model_params,
    data_iterator,
    LEARNING_RATE,
    N_EPOCHS)

nn.save(MODEL_OUT_DIR, "sq_fno.eqx", sq_fno_hyperparams, model)

# Delete Data Pipeline
del data_pipeline