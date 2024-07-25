# from __future__ import annotations
import equinox as eqx
from typing import Callable
import jax
from .SpectralConvolution import SpectralConvolution

class FurierLayer(eqx.Module):
    spectral_conv : SpectralConvolution
    bypass_conv : eqx.nn.Conv
    activation : Callable

    def __init__(
            self, 
            modes : int,
            n_channels : int,
            activation: Callable,
            key):
    
        self.activation = activation

        k1, k2 = jax.random.split(key)
        self.spectral_conv = SpectralConvolution(
            modes=modes,
            n_channels=n_channels, 
            key=k1)
        
        self.bypass_conv = eqx.nn.Conv(
            num_spatial_dims = 3,
            in_channels = n_channels,
            out_channels = n_channels,
            kernel_size = 1,
            padding = 'SAME',
            padding_mode = 'CIRCULAR',
            key = k2)
        
    def __call__(self, x):
        y = self.spectral_conv(x)
        # z = self.bypass_conv(x)
        return self.activation(y)