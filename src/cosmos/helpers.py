import jax

def compute_overdensity(rho : jax.Array) -> jax.Array:
    return (rho - rho.mean()) / rho.mean()