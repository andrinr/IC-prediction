# Implementation of Fast Furier Convolution
# from __future__ import annotations
import equinox as eqx
from typing import Callable
import jax
import jax.numpy as jnp
from typing import Callable
from .fourier_layer import FourierLayer

class FNO(eqx.Module):
    """
    Paper by Li et. al:
    FOURIER NEURAL OPERATOR FOR PARAMETRIC PARTIAL DIFFERENTIAL EQUATIONS

    Implementation inspired by:
    Felix Köhler : https://github.com/Ceyron/machine-learning-and-simulation/
    NeuralOperator: https://github.com/neuraloperator/neuraloperator
    """
    
    lift : eqx.nn.Conv
    furier_layers : list[FourierLayer]
    project : eqx.nn.Conv

    def __init__(
            self,
            modes : int,
            input_channels : int,
            hidden_channels : int,
            output_channels : int,
            activation: Callable,
            n_furier_layers : int,
            key):
         
        k1, k2, k3 = jax.random.split(key, 3)

        self.lift = eqx.nn.Conv(
            num_spatial_dims = 3,
            in_channels = input_channels,
            out_channels = hidden_channels, 
            kernel_size = 3,
            padding = 'SAME',
            padding_mode = 'CIRCULAR',
            key=k1)
        
        self.furier_layers = []
        furier_keys = jax.random.split(k2, n_furier_layers)

        for i in range(n_furier_layers):
            self.furier_layers.append(FourierLayer(
                modes = modes,
                n_channels = hidden_channels,
                activation = activation,
                key = furier_keys[i]))
            
        self.project = eqx.nn.Conv(
            num_spatial_dims = 3,
            in_channels = hidden_channels,
            out_channels = output_channels,
            kernel_size = 3,
            padding = 'SAME',
            padding_mode = 'CIRCULAR',
            key=k3)

        return

    def __call__(self, x : jax.Array):
            
        x = self.lift(x)

        for layer in self.furier_layers:
            x = layer(x)

        x = self.project(x)

        return x