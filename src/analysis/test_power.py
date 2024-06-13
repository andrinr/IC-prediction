import jax.numpy as jnp
import jax
from typing import Tuple
import analysis as an

def gen_data(N):
    key = jax.random.PRNGKey(20)

    rho = jax.random.normal(key, (N, N, N), dtype=jnp.float32) + 1.0
    delta = an.compute_overdensity(rho)

    return delta

def test_power():
    N = 128
    n_bins = 32
    power_spectrum = an.PowerSpectrum(N = N, n_bins = n_bins)

    delta = gen_data(N)

    # print some info about delta
    print(f"delta mean: {jnp.mean(delta)}, delta std: {jnp.std(delta)}, delta min: {jnp.min(delta)}, delta max: {jnp.max(delta)}")
    
    k, P_k = power_spectrum(delta)

    assert k.shape == (n_bins,)
    assert P_k.shape == (n_bins,)

    assert jnp.all(P_k >= 0)

    print(P_k)

    # assert sum of P_k is about n_bins
    assert jnp.allclose(jnp.sum(P_k), n_bins)

    # assert all close to one
    assert jnp.allclose(P_k, 1.0)