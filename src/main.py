import jax.numpy as jnp
import jax
from jax.lib import xla_bridge

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

from CosmoVoxels import CosmoVoxels

print("Jax backend is using %s" % xla_bridge.get_backend().platform)

jax.config.update("jax_enable_x64", False)
jax.config.update("jax_disable_jit", False)

dataset = CosmoVoxels(
    "/shares/feldmann.ics.mnf.uzh/Andrin/IC_GEN/grid/"
)