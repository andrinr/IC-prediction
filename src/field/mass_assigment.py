import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnums=(2, 3))
def linear_ma(
        pos : jnp.ndarray, 
        weight : jnp.ndarray, 
        grid_size : int, 
        dx : float) -> jnp.ndarray:

    coords = jnp.linspace(start=0, stop=1, num=grid_size+1)

    grid = jnp.zeros((grid_size, grid_size, grid_size))

    # find position on the grid
    x_idx = jnp.digitize(pos[0] % 1.0, coords, right=False) - 1
    y_idx = jnp.digitize(pos[1] % 1.0, coords, right=False) - 1
    z_idx = jnp.digitize(pos[2] % 1.0, coords, right=False) - 1

    # assign the mass
    grid = grid.at[x_idx, y_idx, z_idx].add(weight / dx**3)

    return grid