import jax
import jax.numpy as jnp

def central_difference(
        field : jnp.ndarray,
        axis : int, 
        delta : float) -> jnp.ndarray:
    
    field_r = jnp.roll(field, 1, axis=axis)
    field_l = jnp.roll(field, -1, axis=axis)
    
    return (field_r - 2 * field + field_l) / (delta ** 2)

def gradient(
        field : jnp.ndarray,
        delta : float) -> jnp.ndarray:
    
    grad_x = central_difference(field, 0, delta)
    grad_y = central_difference(field, 1, delta)
    grad_z = central_difference(field, 2, delta)

    return jnp.stack([grad_x, grad_y, grad_z], axis=0)