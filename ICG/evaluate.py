# JAX 
import jax.numpy as jnp
import jax
from jax.lib import xla_bridge
# NVIDIA Dali
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.jax import DALIGenericIterator
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

# JAX Settings / Device Info
print("Jax backend is using %s" % xla_bridge.get_backend().platform)
jax.config.update("jax_enable_x64", False)
jax.config.update("jax_disable_jit", False)

# Data Pipeline
dataset = data.VolumetricSequence(
    BATCH_SIZE, INPUT_GRID_SIZE, DATA_DIR, (0, 50), False)

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

model = nn.load(
    MODEL_OUT_DIR, "fno.eqx", nn.fno.FNO, jax.nn.relu)

data = next(data_iterator)
x = jax.device_put(data['end'], jax.devices('gpu')[0])
y_star = jax.device_put(data['start'], jax.devices('gpu')[0])
y = model(x[0])

visualize.compare(
    "compare.jpg", x=x, y=y, y_star=y_star)

# Delete Data Pipeline
del data_pipeline