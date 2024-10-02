import jax
import jax.numpy as jnp
from typing import Tuple
from .growth_factor import compute_growth_factor

def compute_overdensity_mean(rho : jax.Array) -> Tuple[jax.Array, float]:
    """
    Overdensity (delta) of a density field (rho) as defined in cosmology
    """
    mean = rho.mean()
    return (rho - mean) / mean, mean

def compute_overdensity(rho : jax.Array) -> jax.Array:
    """
    Overdensity (delta) of a density field (rho) as defined in cosmology
    """
    mean = rho.mean()
    return (rho - mean) / mean

def compute_rho(overdensity : jax.Array, mean : float) -> jax.Array:
    """
    Get density (rho) from overdensity (delta)
    """
    return overdensity * mean + mean

def normalize(rho : jax.Array, a : float, type : str) -> Tuple[jax.Array, jax.Array]:

    if type == "norm_log_delta_one":
        normalized, attributes = norm_log_delta_one(rho, a)
    else:
        raise NotImplementedError(f"Norm function {type} not found")

    return normalized, attributes

def normalize_inv(norm : jax.Array, attributes : jax.Array, type : str) -> jax.Array:
    if type == "norm_log_delta_one":
        rho = norm_log_delta_one_inv(norm, attributes)
    else:
        raise NotImplementedError(f"Norm function {type} not found")
    
    return rho

def norm_log_delta_one(rho : jax.Array, a : float) -> Tuple[jax.Array, jax.Array]:

    delta, mean = compute_overdensity_mean(rho)

    normalized = jnp.log10((delta / a) + 1)

    attributes = jnp.array([mean, a])

    return normalized, attributes

def norm_log_delta_one_inv(normalized : jax.Array, attributes : jax.Array) -> jax.Array:
    mean = attributes[0]
    a = attributes[1]

    rho = jnp.power(10, (normalized - 1) * a)

    rho = compute_rho(rho, mean)

    return rho