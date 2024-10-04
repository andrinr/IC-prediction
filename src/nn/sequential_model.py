# Implementation of Fast Furier Convolution
# from __future__ import annotations
import equinox as eqx
from typing import Callable
import jax
from typing import Callable
import jax.numpy as jnp
from cosmos import normalize_inv, compute_overdensity, Potential
from functools import partial
from field import gradient


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
            sequential_mode : bool,
            add_potential : bool):
        """
        shape of x:
        [Frames, Channels, Depth, Height, Width]
        """
        f, c, d, h, w = x.shape
        y = jnp.zeros((f-1, c, d, h, w))
        potential = Potential(d)

        # potential_fn = cosmos.Potential(d)
        # time_grid = jnp.ones((1, d, w, h))

        if self.sequential_skip_channels > 0:
            secondary_carry = jnp.ones((self.sequential_skip_channels, d, w, h))

        if sequential_mode:

            
            if self.sequential_skip_channels > 0:
                distribution = jnp.concatenate((x[0], secondary_carry), axis=0)
            else:
                distribution = x[0]

            for i in range(self.sequence_length):
                # potential = potential_fn(carry)

                distribution = self.model[i](distribution) if self.unique_networks else self.model(distribution)
                y = y.at[i].set(distribution[0:1])
                
        else:
            
            # normalize_inv_map = jax.vmap(normalize_inv)
            # rho_sim = normalize_inv_map(x, attributes[:, 0], attributes[:, 0])
            # delta_sim = 

            for i in range(self.sequence_length):
                # potential = potential_fn(x[i])
                
                if add_potential and self.sequential_skip_channels == 0:
                    rho = normalize_inv(x[i], attributes[i], type="log_growth")
                    delta = compute_overdensity(rho)
                    pot = potential(delta)
                    # grad = gradient(pot, 1)
                    distribution = jnp.concatenate((x[i], pot), axis=0)

                elif self.sequential_skip_channels > 0 and not add_potential:
                    distribution = jnp.concatenate((x[i], secondary_carry), axis=0)

                elif add_potential and self.sequential_skip_channels > 0:
                    rho = normalize_inv(x[i], attributes[i], type="log_growth")
                    delta = compute_overdensity(rho)
                    pot = potential(delta)
                    # grad = gradient(pot, 1)
                    distribution = jnp.concatenate((x[i], pot, secondary_carry), axis=0)

                else:
                    distribution = x[i]

                prediction = self.model[i](distribution) if self.unique_networks else self.model(distribution)
                
                y = y.at[i].set(prediction[0:1])

                if self.sequential_skip_channels > 0:
                    secondary_carry = prediction[1:]

        return y