# Implementation of Fast Furier Convolution
# from __future__ import annotations
import equinox as eqx
from typing import Callable
import jax
from typing import Callable
from .fno import FNO

class SequentialFNO(eqx.Module):
    """
    Paper by Li et. al:
    FOURIER NEURAL OPERATOR FOR PARAMETRIC PARTIAL DIFFERENTIAL EQUATIONS

    Implementation inspired by:
    Felix Köhler : https://github.com/Ceyron/machine-learning-and-simulation/
    NeuralOperator: https://github.com/neuraloperator/neuraloperator
    """

    fno : list[FNO]

    def __init__(
            self,
            modes : int,
            hidden_channels : int,
            activation: Callable,
            n_furier_layers : int,
            sequence_length : int,
            key):
        
        keys = jax.random.split(key, sequence_length)

        self.fno = []

        for i in range(sequence_length):
            self.fno.append(
                FNO(modes = modes,
                    hidden_channels = hidden_channels,
                    activation = activation,
                    n_furier_layers=n_furier_layers,
                    key=keys[i]))
        return

    def __call__(self, x : jax.Array, sequence_index : int):
        return self.fno[sequence_index](x)