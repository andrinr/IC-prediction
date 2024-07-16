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
data_callable = CosmosData(8, 128, DATA_ROOT, (0, 30), False)
# print(data_callable({0, 0, 0, 0}))

@pipeline_def(
    batch_size=BATCH_SIZE,
    num_threads=2, 
    device_id=0,
    py_num_workers=4,
    py_start_method="spawn")
def pipeline():
    [grids] = fn.external_source(
        source=data_callable,
        num_outputs=1,
        batch=False,
        dtype=types.FLOAT
    )
    fn.resize(
        grids
    )
    return grids

pipe = pipeline()
pipe.build()
out = pipe.run()

print(out[0])

del pipe