from __future__ import annotations
import equinox as eqx
from typing import Callable
import jax

class DoubleConv(eqx.Module):
    conv_1: eqx.nn.Conv
    conv_2: eqx.nn.Conv
    activation : Callable

    def __init__(
            self, 
            num_spatial_dims: int,
            in_channels: int,
            out_channels: int,
            activation: Callable,
            padding : str,
            padding_mode: str,
            key):
        
        key1, key2 = jax.random.split(key)
        self.conv_1 = eqx.nn.Conv(
            num_spatial_dims, 
            in_channels, 
            out_channels, 
            kernel_size=3, 
            padding=padding,
            padding_mode=padding_mode, 
            key=key1)
        
        self.conv_2 = eqx.nn.Conv(
            num_spatial_dims, 
            out_channels, 
            out_channels, 
            kernel_size=3, 
            padding=padding,
            padding_mode=padding_mode, 
            key=key2)
        
        self.activation = activation
        
    def __call__(self, x):
        x = self.conv_1(x)
        x = self.activation(x)
        x = self.conv_2(x)
        x = self.activation(x)
        return x