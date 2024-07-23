# Implementation of Fast Furier Convolution
from __future__ import annotations
import equinox as eqx
from typing import Callable
import jax
import jax.numpy as jnp
from typing import Callable
from .DoubleConv import DoubleConv

class FFC(eqx.Module):
    """
    All variables ending with _fs are in furier space, others in normal physical space.
    """

    lift : DoubleConv
    drop : DoubleConv

    def __init__(
            self,
            rnd_key,
            hidden_channels : int,
            n_grid : int,
            activation: Callable):

        self.n_grid = n_grid

        keys = jax.random.split(rnd_key, 4)

        self.lift = DoubleConv(
            num_spatial_dims = 3,
            in_channels = 1,
            out_channels = hidden_channels,
            activation = activation,
            padding = 'SAME',
            padding_mode = 'CIRCULAR',
            key = keys[0])
        
        self.drop = DoubleConv(
            num_spatial_dims = 3,
            in_channels = hidden_channels,
            out_channels = 1,
            activation = activation,
            padding = 'SAME',
            padding_mode = 'CIRCULAR',
            key = keys[1])

        return

    def to_fs(x : jnp.ndarray):
        x_fs = jnp.fft.fftn(x)
        x_fs = jnp.fft.fftshift(x)
        return jnp.concatenate([x_fs.real(), x_fs.complex()])

    def from_fs(x_fs : jnp.ndarray):
        x_fs = jnp.fft.ifftshift(x_fs)
        return jnp.fft.ifftn(x_fs)

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


