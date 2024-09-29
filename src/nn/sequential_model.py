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

    model : eqx.Module | list[eqx.Module]
    unique_networks : bool
    sequence_length : int

    def __init__(
            self,
            sequence_length : int,
            constructor : eqx.Module,
            parameters : dict,
            activation: Callable,
            unique_networks : bool,
            key):
        
        self.sequence_length = sequence_length

        self.unique_networks = unique_networks
        if not unique_networks:
            self.model = constructor(key=key, activation=activation, **parameters)
        else:
            self.model = []
            for i in range(sequence_length):
                self.model.append(constructor(key=key, activation=activation, **parameters))
      
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

        # potential_fn = cosmos.Potential(d)
        # time_grid = jnp.ones((1, d, w, h))

        secondary_carry = jnp.ones((1, d, w, h))

        if sequential_mode:
            carry = jnp.concatenate(jnp.array([x[0], secondary_carry]))
            for i in range(self.sequence_length):
                # potential = potential_fn(carry)
                carry = self.model[i](carry) if self.unique_networks else self.model(carry)
                y = y.at[i].set(carry[0:1])
                

        else:
            for i in range(self.sequence_length):
                # potential = potential_fn(x[i])
                print(x[i].shape)
                print(secondary_carry.shape)
                carry = jnp.concatenate(jnp.array([x[i], secondary_carry]))
                carry = self.model[i](carry) if self.unique_networks else self.model(carry)
                
                y = y.at[i].set(carry[0:1])

                secondary_carry = carry[1:2]

        return y