import jax.numpy as jnp
import jax

def normalize(rho : jax.Array):

    rho /= 10**4
    rho += 1
    rho = jnp.log10(rho)
    min = rho.min()
    max = rho.max()

    rho -= min
    rho /= max

    return rho, min, max

def normalize_inv(rho_normalized : jax.Array, min : float, max : float):

    rho = rho_normalized * max
    rho += min

    rho = jnp.power(10, rho)
    rho -= 1
    rho *= 10**4

    return rho
