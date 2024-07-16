# JAX 
import jax.numpy as jnp
import jax
from jax.lib import xla_bridge
# Other
import os
# Local
from CosmoData import CosmosData

DATA_ROOT = "/shares/feldmann.ics.mnf.uzh/Andrin/IC_GEN/grid/"
GRID_SIZE=32

print("Jax backend is using %s" % xla_bridge.get_backend().platform)

jax.config.update("jax_enable_x64", False)
jax.config.update("jax_disable_jit", False)

# iterate over folders in DATA_ROOT
data_iterator = CosmosData(
    8, 128, DATA_ROOT, (0, 30), False)

for batch in data_iterator:
    print(".")