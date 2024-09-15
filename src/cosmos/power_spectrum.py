import jax.numpy as jnp
import jax
from typing import Tuple
from .frequency_operation import FrequencyOperation

class PowerSpectrum(FrequencyOperation):
    n_bins : int
    index_grid : jax.Array
    n_modes : jax.Array

    def __init__(self, n_grid : int, n_bins : int):
        super().__init__(n_grid=n_grid)
        self.n_bins = n_bins

        self.index_grid = jnp.digitize(self.k, jnp.linspace(0, self.nyquist, self.n_bins), right=True) - 1

        self.n_modes = jnp.zeros(self.n_bins)
        self.n_modes = self.n_modes.at[self.index_grid].add(1)

    def __call__(self, delta : jax.Array) -> Tuple[jax.Array, jax.Array]:
        # get the density field in furier space
        delta_k = jnp.fft.rfftn(delta)

        # Remove negative wave numbers due to Nyquist folding
        # TODO: ensure this is correct?
        # delta_k = delta_k[:self.nyquist:, :self.nyquist:, :self.nyquist]

        print(delta_k.shape)
        print(self.k.shape)
        print(self.index_grid.shape)
        
        power = jnp.zeros(self.n_bins)
        power = power.at[self.index_grid].add(jnp.abs(delta_k))

        # compute the average power
        power = power / self.n_modes
        power = power / self.N ** 2 

        power = jnp.where(jnp.isnan(power), 0, power)

        k = jnp.linspace(0, self.nyquist, self.n_bins)

        return k, power

