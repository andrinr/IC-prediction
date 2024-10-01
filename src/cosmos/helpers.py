import jax
from typing import Tuple

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

def compute_rho(overdensity : jax.Array, mean : float) -> Tuple[jax.Array]:
    """
    Get density (rho) from overdensity (delta)
    """
    return overdensity * mean + mean