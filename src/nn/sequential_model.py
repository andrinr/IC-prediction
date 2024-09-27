# Implementation of Fast Furier Convolution
# from __future__ import annotations
import equinox as eqx
from typing import Callable
import jax
from typing import Callable
import jax.numpy as jnp
import cosmos

class SequentialModel(eqx.Module):
    """
    Sequential Fourier Neural Operator
    """

    model : eqx.Module
    sequence_length : int
    normalization_params : jax.Array

    def __init__(
            self,
            sequence_length : int,
            constructor : eqx.Module,
            parameters : dict,
            activation: Callable,
            key):
        
        self.sequence_length = sequence_length

        self.model = constructor(key=key, activation=activation, **parameters)

        self.normalization_params = jnp.zeros(2)
        self.normalization_params = self.normalization_params.at[0].set(1)
        self.normalization_params = self.normalization_params.at[1].set(1)
                
        return
    
    def normalize(self, x : jax.Array):
        # return jnp.log10(x / (self.normalization_params[0]*10**4) + self.normalization_params[1])
        return x

    def __call__(self, x : jax.Array, sequential_mode : bool):
        """
        shape of x:
        [Frames, Channels, Depth, Height, Width]
        """
        f, c, d, h, w = x.shape
        y = jnp.zeros((f-1, c, d, h, w))

        potential_fn = cosmos.Potential(d)
        time_grid = jnp.ones((1, d, w, h))

        if sequential_mode:
            carry = x[0]
            for i in range(self.sequence_length):
                # potential = potential_fn(carry)
                x_ = jnp.concatenate([carry, time_grid * i/self.sequence_length + 1], axis=0)
                carry = self.model(x_)
                y = y.at[i].set(carry)

        else:
            for i in range(self.sequence_length):
                # potential = potential_fn(x[i])
                x_ = jnp.concatenate([x[i], time_grid * i/self.sequence_length + 1], axis=0)
                print(self.normalize(x[i]).shape)
                y = y.at[i].set(self.model(x_))

        return y