# JAX 
import jax
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
    MODEL_OUT_DIR, "fno.eqx", nn.fno.FNO, jax.nn.relu)
model_params, model_static = eqx.partition(model, eqx.is_array)

data = next(data_iterator)
sequence = jax.device_put(data['sequence'], jax.devices('gpu')[0])[0]
n_frames = sequence.shape[0]
sequence_prediction = nn.predict_sequence(
    sequence[0], 
    steps = n_frames - 1,
    model_params = model_params, 
    model_static = model_static)

timeline = data["time"][0]
print(timeline)
print(sequence.shape)
print(sequence_prediction.shape)
print(jax.numpy.sum(jax.numpy.isnan(sequence_prediction)))

visualize.sequence(
    "seq.jpg", 
    sequence = sequence, 
    sequence_prediction = sequence_prediction,
    timeline = timeline)

# Delete Data Pipeline
del data_pipeline