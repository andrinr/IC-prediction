# JAX 
import jax
import jax.numpy as jnp
# NVIDIA Dali
from nvidia.dali.plugin.jax import DALIGenericIterator
# Equinox
import equinox as eqx
# Local
import nn
import data
import visualize

# Parameters
DATA_DIR = "/shares/feldmann.ics.mnf.uzh/Andrin/IC_GEN/grid/"
MODEL_OUT_DIR = "ICG/model_params/"
INPUT_GRID_SIZE = 128
GRID_SIZE = 64
BATCH_SIZE = 8

# Data Pipeline
dataset = data.VolumetricSequence(
    grid_size = INPUT_GRID_SIZE,
    directory = DATA_DIR,
    start = 0,
    steps = 50,
    stride = 10,
    flip=True)

data_pipeline = data.volumetric_sequence_pipe(dataset, GRID_SIZE)
data_iterator = DALIGenericIterator(data_pipeline, ["sequence", "time"])

model = nn.load(
    MODEL_OUT_DIR, "sq_fno.eqx", nn.FNO, jax.nn.relu)

data = next(data_iterator)
sequence = jax.device_put(data['sequence'], jax.devices('gpu')[0])[0]
pred = jnp.zeros_like(sequence)

n_frames = sequence.shape[0]
pred = pred.at[0].set(sequence[0])
for i in range(1, n_frames):
    print(i)
    pred = pred.at[i].set(model(pred[i-1]))

timeline = data["time"][0]

visualize.sequence(
    "seq.jpg", 
    sequence = sequence, 
    sequence_prediction = pred,
    timeline = timeline)

# Delete Data Pipeline
del data_pipeline