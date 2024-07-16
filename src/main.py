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
# Local
from VolumetricSequence import VolumetricSequence

DATA_ROOT = "/shares/feldmann.ics.mnf.uzh/Andrin/IC_GEN/grid/"
INPUT_GRID_SIZE = 128
GRID_SIZE = 32
BATCH_SIZE = 4

print("Jax backend is using %s" % xla_bridge.get_backend().platform)

jax.config.update("jax_enable_x64", False)
jax.config.update("jax_disable_jit", False)

# iterate over folders in DATA_ROOT
data_callable = VolumetricSequence(BATCH_SIZE, INPUT_GRID_SIZE, DATA_ROOT, (0, 30), False)
# print(data_callable({0, 0, 0, 0}))

@pipeline_def(
    batch_size=BATCH_SIZE,
    num_threads=2, 
    device_id=0,
    py_num_workers=4,
    py_start_method="spawn")
def pair_pipeline():
    [start, end] = fn.external_source(
        source=data_callable,
        num_outputs=2,
        batch=False,
        dtype=types.FLOAT
    )

    start = fn.resize(
        start,
        interp_type = types.INTERP_CUBIC,
        antialias=False,
        resize_x = GRID_SIZE,
        resize_y = GRID_SIZE,
        resize_z = GRID_SIZE,
    )

    end = fn.resize(
        start,
        interp_type = types.INTERP_CUBIC,
        antialias=False,
        resize_x = GRID_SIZE,
        resize_y = GRID_SIZE,
        resize_z = GRID_SIZE,
    )

    return start, end

pipe = pair_pipeline()
pipe.build()
out = pipe.run()

print(out[0])

del pipe