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
from CosmoData import CosmosData

DATA_ROOT = "/shares/feldmann.ics.mnf.uzh/Andrin/IC_GEN/grid/"
GRID_SIZE = 32
BATCH_SIZE = 8

print("Jax backend is using %s" % xla_bridge.get_backend().platform)

jax.config.update("jax_enable_x64", False)
jax.config.update("jax_disable_jit", False)

# iterate over folders in DATA_ROOT
data_callable = CosmosData(
    8, 128, DATA_ROOT, (0, 30), False)

@pipeline_def(BATCH_SIZE, num_threads=2, device_id=0)
def pipeline():
    grids = fn.external_source(
        source=data_callable,
        num_outputs=2,
        batch=False,
        dtype=types.FLOAT32
    )
    return grids

pipe = pipeline()
pipe.build()
out = pipe.run()