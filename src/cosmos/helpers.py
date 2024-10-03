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

    if type == "log_growth":
        normalized, attributes = norm_log_growth(rho, a)
    elif type == "delta":
        normalized, attributes = norm_delta(rho)
    elif type == "ssm":
        normalized, attributes = norm_ssm(rho)
    else:
        raise NotImplementedError(f"Norm function {type} not found")

    return normalized, attributes

def normalize_inv(norm : jax.Array, attributes : jax.Array, type : str) -> jax.Array:
    if type == "log_growth":
        rho = norm_log_growth_inv(norm, attributes)
    elif type == "delta":
        rho = norm_delta_inv(norm, attributes)
    elif type == "ssm":
        rho = norm_ssm_inv(norm, attributes)
    else:
        raise NotImplementedError(f"Norm function {type} not found")
    
    return rho

def norm_ssm(rho : jax.Array):
    mean = rho.mean()
    var = rho.var()

    norm = rho - mean 
    norm /= var

    attributes = jnp.array([mean, var])

    return norm, attributes

def norm_ssm_inv(norm : jax.Array, attributes : jax.Array):

    mean = attributes[0]
    var = attributes[1]

    rho = norm * var
    rho += mean 

    return rho

def norm_delta(rho : jax.Array):
    delta, mean = compute_overdensity_mean(rho)

    attributes = jnp.array([mean])

    return delta, attributes

def norm_delta_inv(norm : jax.Array, attributes : jax.Array):

    rho = compute_rho(norm, mean=attributes[0])

    return rho

def norm_log_growth(rho : jax.Array, a : float) -> Tuple[jax.Array, jax.Array]:

    normalized = rho/10**2
    normalized += 1.5
    normalized /= a
    normalized = jnp.log10(normalized)

    attributes = jnp.array([a])

    return normalized, attributes

def norm_log_growth_inv(normalized : jax.Array, attributes : jax.Array) -> jax.Array:
    a = attributes[0]

    rho = jnp.power(10, normalized)
    rho *= a
    rho -= 1.5
    rho *= 10**2

    # rho = compute_rho(rho, mean)

    return rho