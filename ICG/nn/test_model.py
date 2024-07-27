# from __future__ import annotations
import equinox as eqx
from typing import Callable
import jax
import jax.numpy as jnp
    
class Dummy(eqx.Module):
    lift : eqx.nn.Conv
    projection : eqx.nn.Conv
    activation : Callable

    def __init__(
            self,
            num_spatial_dims: int,
            channels : int,
            activation: Callable,
            padding: str,
            padding_mode: str,
            key):

        self.activation = activation

        self.lift = eqx.nn.Conv(
            in_channels=channels,
            out_channels=channels * 8,
            num_spatial_dims=num_spatial_dims,
            kernel_size=1,
            padding_mode=padding_mode,
            padding=padding,
            key=key)
        
        self.projection = eqx.nn.Conv(
            in_channels=channels * 8,
            out_channels=channels,
            num_spatial_dims=num_spatial_dims,
            kernel_size=1,
            padding_mode=padding_mode,
            padding=padding,
            key=key)

    def __call__(self, x):
        
        x = self.lift(x)
        x = self.projection(x)

        return x