# Implementation of Fast Furier Convolution

import equinox as eqx
from typing import Callable
import jax
import jax.numpy as jnp
from typing import Callable

class FFC(eqx.Module):
    """
    All variables ending with _fs are in furier space, others in normal physical space.
    """

    hidden_channels : int
    n_grid : int
    n_grid_2 : int
    n_grid_4 : in
    activation_function : Callable
    start : eqx.nn.Conv
    start_fs : eqx.nn.Conv
    middle : list[eqx.nn.Conv]
    middle_fs : list[eqx.nn.Conv]
    end : eqx.nn.Conv
    end_fs : eqx.nn.Conv

    def __init__(
            self,
            rnd_key,
            n_grid : int,
            hidden_channels : int,
            activation: Callable):

        self.n_grid = n_grid
        self.n_grid_2 = n_grid // 2
        self.n_grid_4 = n_grid // 4
        self.hidden_channels = hidden_channels

        self.start = eqx.nn.Conv(
            num_spatial_dims = 3,
            in_channels = 1,
            out_channels = hidden_channels)
        
        self.start_fs = eqx.nn.Conv(
            num_spatial_dims = 3,
            in_channels = 1,
            out_channels = hidden_channels)

        self.end = 




        
        return

    def to_fs(x):
        x = jnp.fft.fftshift(x)
        return jnp.fft.fftn(x)

    def from_fs(x):
        x = ifftshift(x)
        return jnp.fft.ifftn(x)

    def __call__(self, x):

        x_fs = self.to_fs(x)
        x_fs = x_fs[n_grid_4:n_grid_4, n_grid_4:n_grid_4, n_grid_4:n_grid_4]

        x = self.start(x)
        x_fs = self.start_fs(x_fs)

        x = jnp.concatenate([x, self.from_fs(x_fs)], axis=0)
        x_fs = jnp.concatenate([x_fs, self.to_fs(x)], axis=0)

        # repeat the above some times

        x = eqx.nn.Conv(
            num_spatial_dims = 3,
            in_channels = hidden_channels,
            out_channels = 1)

        return x


