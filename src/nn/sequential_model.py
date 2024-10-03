# Implementation of Fast Furier Convolution
# from __future__ import annotations
import equinox as eqx
from typing import Callable
import jax
from typing import Callable
import jax.numpy as jnp
from cosmos import compute_overdensity
from functools import partial

class SequentialModel(eqx.Module):
    """
    Sequential Fourier Neural Operator
    """

    model : eqx.Module | list[eqx.Module]
    unique_networks : bool
    sequence_length : int
    sequential_skip_channels : int

    def __init__(
            self,
            sequence_length : int,
            constructor : eqx.Module,
            parameters : dict,
            unique_networks : bool,
            sequential_skip_channels : int,
            key):
        
        self.sequence_length = sequence_length

        self.unique_networks = unique_networks
        if not unique_networks:
            self.model = constructor(key=key, **parameters)
        else:
            self.model = []
            for i in range(sequence_length):
                self.model.append(constructor(key=key, **parameters))

        self.sequential_skip_channels = sequential_skip_channels
        return
    
    def normalize(self, x : jax.Array):
        # return jnp.log10(x / (self.normalization_params[0]*10**4) + self.normalization_params[1])
        return x

    def __call__(
            self, 
            x : jax.Array, 
            attributes : jax.Array,
            sequential_mode : bool):
        """
        shape of x:
        [Frames, Channels, Depth, Height, Width]
        """
        f, c, d, h, w = x.shape
        y = jnp.zeros((f-1, c, d, h, w))

        # potential_fn = cosmos.Potential(d)
        # time_grid = jnp.ones((1, d, w, h))

        if self.sequential_skip_channels > 0:
            secondary_carry = jnp.ones((self.sequential_skip_channels, d, w, h))

        if sequential_mode:
            if self.sequential_skip_channels > 0:
                carry = jnp.concatenate((x[0], secondary_carry), axis=0)
            else:
                carry = x[0]

            for i in range(self.sequence_length):
                # potential = potential_fn(carry)
                carry = self.model[i](carry) if self.unique_networks else self.model(carry)
                y = y.at[i].set(carry[0:1])
                
        else:
            
            # normalize_inv_map = jax.vmap(normalize_inv)
            # rho_sim = normalize_inv_map(x, attributes[:, 0], attributes[:, 0])
            # delta_sim = 

            for i in range(self.sequence_length):
                # potential = potential_fn(x[i])

                if self.sequential_skip_channels > 0:
                    print(x[i].shape)
                    print(secondary_carry.shape)
                    carry = jnp.concatenate((x[i], secondary_carry), axis=0)
                else:
                    carry = x[i]

                carry = self.model[i](carry) if self.unique_networks else self.model(carry)
                
                y = y.at[i].set(carry[0:1])

                secondary_carry = carry[1:]

        return y