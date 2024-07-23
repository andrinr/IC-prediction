from __future__ import annotations
import equinox as eqx
from typing import Callable
import jax
import jax.numpy as jnp

class SpectralConvolution(eqx.Module):
    """
    Paper by Li et. al:
    FOURIER NEURAL OPERATOR FOR PARAMETRIC PARTIAL DIFFERENTIAL EQUATIONS

    Implementation inspired by:
    Felix Köhler : https://github.com/Ceyron/machine-learning-and-simulation/
    """

    modes : int
    activation : Callable
    weights_real : list[jax.Array]
    weights_imag : list[jax.Array]

    def __init__(
            self, 
            modes : int,
            n_channels : int,
            activation: Callable,
            key):
        
        self.modes = modes
        self.activation = activation
        keys = jax.random.split(key, 8)
        
        scale = 1.0 / (n_channels ** 2)

        self.weights_real = []
        self.weights_imag = []

        for i in range(4):
            real = jax.random.uniform(
                keys[i], 
                (n_channels, n_channels, modes, modes, modes, 2),
                minval=-scale, maxval=scale)
            
            imag = jax.random.uniform(
                keys[i + 4], 
                (n_channels, n_channels, modes, modes, modes, 2),
                minval=-scale, maxval=scale)
            
            self.weights_real.append(real)
            self.weights_imag.append(imag)
        
    def complex_mul3d(self, a, b):
        op = jnp.einsum("ixyz,ioxyz->oxyz")

        return op(a, b)

    def __call__(self, x):
        # x shape : n_channels, N, N, N
        # x_fs shape : n_channels, N, N, N // 2, 2
        print(x.shape)

        x_fs = jnp.fft.rfftn(x)

        print(x_fs.shape)

        out_fs = jnp.zeros_like(x_fs)

        weights = self.weights_real[0] + 1j * self.weights_imag[0]
        out_fs = out_fs.at[:, :self.modes, :self.modes, :self.modes].set(
            self.complex_mul(weights, x_fs[:, :self.modes, :self.modes, :self.modes]))
        
        weights = self.weights_real[1] + 1j * self.weights_imag[1]
        out_fs = out_fs.at[:, -self.modes:, :self.modes, :self.modes].set(
            self.complex_mul(self.weights, x_fs[:, -self.modes:, :self.modes, :self.modes]))
        
        weights = self.weights_real[2] + 1j * self.weights_imag[2]
        out_fs = out_fs.at[:, :self.modes, -self.modes:, :self.modes].set(
            self.complex_mul(weights, x_fs[:, :self.modes, -self.modes:, :self.modes]))
        
        weights = self.weights_real[3] + 1j * self.weights_imag[3]
        out_fs = out_fs.at[:, :-self.modes, :-self.modes, :self.modes].set(
            self.complex_mul(weights, x_fs[:, :-self.modes, :-self.modes, :self.modes]))
           
        out = jax.vmap(jnp.fft.ifftn)(out_fs)

        print(out.shape)

        return out