# Implementation of Fast Furier Convolution

import equinox as eqx
from typing import Callable
import jax
import jax.numpy as jnp

class FFC(eqx.Module):

    def __init__(
            self,
            rnd_key):
        
        return
        

    def __call__(self, x):

        x_furier = jnp.fft.fftn(x)

        # apply some convolutions to x and x_furier

        

