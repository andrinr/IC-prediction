import jax.numpy as jnp

def compute_overdensity(rho : jnp.ndarray) -> jnp.ndarray:
    return (rho - rho.mean()) / rho.mean()