from __future__ import annotations
import equinox as eqx
from typing import Callable
import jax
import jax.numpy as jnp

class FurierLayer(eqx.Module):
    modes : int
    activation : Callable
    rl_lin : list[eqx.nn.Linear]
    im_lin : list[eqx.nn.Linear]

    def __init__(
            self, 
            modes : int,
            n_channels : int,
            activation: Callable,
            padding : str,
            padding_mode: str,
            key):
        
        self.modes = modes
        keys = jax.random.split(key, 8)
        self.rl_lin = []
        self.im_lin = []

        for i in range(4):
            real_linear = eqx.nn.Linear(
                n_channels * modes ** 3,
                n_channels * modes ** 3, 
                key=keys[2 * i])
            
            imag_linear = eqx.nn.Linear(
                n_channels * modes ** 3,
                n_channels * modes ** 3, 
                key=keys[2 * i + 1])
            
            self.rl_lin.append(real_linear)
            self.im_lin.append(imag_linear)

        self.activation = activation
        
    def __call__(self, x):
        
        x_fs = jnp.fft.fftn(x)

        out_fs = jnp.zeros_like(x_fs)

        out_fs = out_fs.at[:self.modes, :self.modes, :self.modes, 0].set(
            self.rl_lin[0](x_fs[:self.modes, :self.modes, :self.modes, 0]))
        out_fs = out_fs.at[:self.modes, :self.modes, :self.modes, 1].set(
            self.im_lin[0](x_fs[:self.modes, :self.modes, :self.modes, 1]))
        
        out_fs = out_fs.at[:-self.modes, :self.modes, self.modes:, 0].set(
            self.rl_lin[1](x_fs[:-self.modes, :self.modes, self.modes:, 0]))
        out_fs = out_fs.at[:-self.modes, :self.modes, self.modes:, 1].set(
            self.im_lin[1](x_fs[:-self.modes, :self.modes, self.modes:, 1]))
        
        out_fs = out_fs.at[:self.modes, :-self.modes, self.modes:, 0].set(
            self.rl_lin[2](x_fs[:self.modes, :-self.modes, self.modes:, 0]))
        out_fs = out_fs.at[:self.modes, :-self.modes, self.modes:, 1].set(
            self.im_lin[2](x_fs[:self.modes, :-self.modes, self.modes:, 1]))
        
        out_fs = out_fs.at[:-self.modes, :-self.modes, :self.modes, 0].set(
            self.rl_lin[3](x_fs[:-self.modes, :-self.modes, :self.modes, 0]))
        out_fs = out_fs.at[:-self.modes, :-self.modes, :self.modes, 1].set(
            self.im_lin[3](x_fs[:-self.modes, :-self.modes, :self.modes, 1]))
        
        out = jnp.fft.ifftn(out_fs)

        return out