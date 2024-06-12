import jax.numpy as jnp
import jax
from typing import Tuple
from power import PowerSpectrum, compute_overdensity

def test_power():
    N = 128
    n_bins = 32
    power = PowerSpectrum(N, n_bins)

    key = jax.random.PRNGKey(0)

    rho = jax.random.normal(key, (N, N, N))
    delta = compute_overdensity(rho)
    k, p = power(delta)

    assert k.shape == (n_bins,)
    assert p.shape == (n_bins,)

    assert jnp.all(p >= 0)

    print("All tests passed!")
