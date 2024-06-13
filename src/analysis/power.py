import jax.numpy as jnp
from typing import Tuple

def compute_overdensity(rho : jnp.ndarray) -> jnp.ndarray:
    return (rho - rho.mean()) / rho.mean()

class PowerSpectrum:
    N : int
    N_half : int
    n_bins : int
    index_grid : jnp.ndarray

    def __init__(self, N : int, n_bins : int):
        self.N = N
        self.N_half = N//2
        self.n_bins = n_bins

        coords = jnp.linspace(0, self.N_half, self.N_half)
        kx, ky, kz = jnp.meshgrid(coords, coords, coords)
        k = jnp.sqrt(kx**2 + ky**2 + kz**2)

        self.index_grid = jnp.digitize(k, jnp.linspace(0, self.N_half, self.n_bins), right=True) - 1

    def __call__(self, delta : jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        delta_k = jnp.fft.fftn(delta)

        # Remove negative wave numbers due to Nyquist folding
        delta_k = delta_k[:self.N_half:, :self.N_half:, :self.N_half]
        
        power = jnp.zeros(self.n_bins)
        power = power.at[self.index_grid].add(jnp.abs(delta_k))

        # count the number of modes in each bin
        n_modes = jnp.zeros(self.n_bins)
        n_modes = n_modes.at[self.index_grid].add(1)

        print(jnp.sum(n_modes))
        print(self.N_half**3)

        # compute the average power
        # power = power / self.N_half**3
        power = power / n_modes

        power = jnp.where(jnp.isnan(power), 0, power)

        jnp.sum(power)

        k = jnp.linspace(0, self.N_half, self.n_bins)

        return k, power

