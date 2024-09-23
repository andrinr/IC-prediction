# Implementation of Fast Furier Convolution
# from __future__ import annotations
import equinox as eqx
from typing import Callable
import jax
from typing import Callable
import jax.numpy as jnp

class SequentialModel(eqx.Module):
    """
    Sequential Fourier Neural Operator
    """

    models : list[eqx.Module]

    def __init__(
            self,
            sequence_length : int,
            constructor : eqx.Module,
            parameters : dict,
            activation: Callable,
            key):
        
        keys = jax.random.split(key, sequence_length)

        self.models = []

        for i in range(sequence_length):
            self.models.append(
                constructor(key=keys[i], activation=activation, **parameters))
                
        return

    def __call__(self, x : jax.Array, sequential_mode : bool):
        """
        shape of x:
        [Frames, Channels, Depth, Height, Width]
        """
        f, c, d, h, w = x.shape
        y = jnp.zeros((f-1, c, d, h, w))

        if sequential_mode:
            carry = x[0]
            for i, model in enumerate(self.models):
                carry = model(carry)
                y = y.at[i].set(carry)

        else:
            for i, model in enumerate(self.models):
                y = y.at[i].set(model(x[i]))

        return y