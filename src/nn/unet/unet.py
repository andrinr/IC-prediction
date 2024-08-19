# from __future__ import annotations
import equinox as eqx
from typing import Callable
import jax
import jax.numpy as jnp

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
    
class UNet(eqx.Module):
    lifting : DoubleConv
    down_sampling : list[eqx.nn.Conv]
    left_arc : list[eqx.nn.Conv]
    right_arc : list[eqx.nn.Conv]
    up_sampling : list[eqx.nn.Conv]
    projection : eqx.nn.Conv

    def __init__(
            self,
            num_spatial_dims: int,
            in_channels: int,
            out_channels: int,
            hidden_channels: int,
            num_levels: int,
            activation: Callable,
            padding: str,
            padding_mode: str,
            key):

        key, key_liftiing, key_projection = jax.random.split(key, 3)

        self.lifting = DoubleConv(
            num_spatial_dims, 
            in_channels,
            hidden_channels,
            activation,
            padding,
            padding_mode,
            key_liftiing
        )

        self.projection = eqx.nn.Conv(
            num_spatial_dims,
            hidden_channels,
            out_channels,
            kernel_size=1,
            key=key_projection
        )

        channels_per_level = [hidden_channels * 2**i for i in range(num_levels + 1)]

        self.down_sampling = []
        self.left_arc = []
        self.right_arc = []
        self.up_sampling = []

        for (upper_level_channels, lower_level_channels) in zip(
            channels_per_level[:-1], channels_per_level[1:]):
            key, key_down, key_left, key_right, key_up = jax.random.split(key, 5)

            self.down_sampling.append(
                eqx.nn.Conv(
                    num_spatial_dims,
                    upper_level_channels,
                    upper_level_channels,
                    kernel_size=3,
                    stride=2,
                    padding=padding,
                    padding_mode=padding_mode,
                    key=key_down
                )
            )

            self.left_arc.append(
                DoubleConv(
                    num_spatial_dims,
                    upper_level_channels,
                    lower_level_channels,
                    activation,
                    padding,
                    padding_mode,
                    key_left
                )
            )

            self.up_sampling.append(
                eqx.nn.ConvTranspose(
                    num_spatial_dims,
                    lower_level_channels,
                    upper_level_channels,
                    kernel_size=3,
                    stride=2,
                    padding=padding,
                    padding_mode=padding_mode,
                    # output_padding=1,
                    key=key_up
                )
            )

            self.right_arc.append(
                DoubleConv(
                    num_spatial_dims,
                    lower_level_channels,
                    upper_level_channels,
                    activation,
                    padding,
                    padding_mode,
                    key_right
                )
            )

    def __call__(self, x):
        x = self.lifting(x)
        residuals = []

        for down, left in zip(self.down_sampling, self.left_arc):
            residuals.append(x)
            x = down(x)
            x = left(x)

        for right, up in zip(reversed(self.right_arc), reversed(self.up_sampling)):
            x = up(x)
            x = jnp.concatenate([x, residuals.pop()], axis=0)
            x = right(x)

        x = self.projection(x)
        
        return x