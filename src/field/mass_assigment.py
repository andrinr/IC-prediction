import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnums=(2, 3))
def linear_ma(
        pos : jax.Array, 
        weight : jax.Array, 
        grid_size : int, 
        dx : float) -> jax.Array:

    coords = jnp.linspace(start=0, stop=1, num=grid_size+1)

    grid = jnp.zeros((grid_size, grid_size, grid_size))

    # find position on the grid
    x_idx = jnp.digitize(pos[0] % 1.0, coords, right=False) - 1
    y_idx = jnp.digitize(pos[1] % 1.0, coords, right=False) - 1
    z_idx = jnp.digitize(pos[2] % 1.0, coords, right=False) - 1

    # assign the mass
    grid = grid.at[x_idx, y_idx, z_idx].add(weight / dx**3)

    return grid

@partial(jax.jit, static_argnums=(2, 3))
def cic_ma(
        pos : jax.Array, 
        weight : jax.Array, 
        n : int, 
        dx : float) -> jax.Array:
    
    coords = jnp.linspace(start=0, stop=1, num=n+1)

    field = jnp.zeros((n, n, n))

    # find position on the grid
    x = jnp.digitize(pos[0] % 1.0, coords, right=False) - 1
    y = jnp.digitize(pos[1] % 1.0, coords, right=False) - 1
    z = jnp.digitize(pos[2] % 1.0, coords, right=False) - 1

    # find the weights
    xw = (pos[0] % 1.0 - coords[x]) / dx
    yw = (pos[1] % 1.0 - coords[y]) / dx
    zw = (pos[2] % 1.0 - coords[z]) / dx

    weight_ = weight / dx**3

    # assign the mass
    field = field.at[x, y, z].add(weight_ * (1 - xw) * (1 - yw) * (1 - zw))
    field = field.at[(x + 1) % n, y, z].add(weight_ * xw * (1 - yw) * (1 - zw))
    field = field.at[x, (y + 1) % n, z].add(weight_ * (1 - xw) * yw * (1 - zw))
    field = field.at[(x + 1) % n, (y + 1) % n, z].add(weight_ * xw * yw * (1 - zw))
    field = field.at[x, y, (z + 1) % n].add(weight_ * (1 - xw) * (1 - yw) * zw)
    field = field.at[(x + 1) % n, y, (z + 1) % n].add(weight_ * xw * (1 - yw) * zw)
    field = field.at[x, (y + 1) % n, (z + 1) % n].add(weight_ * (1 - xw) * yw * zw)
    field = field.at[(x + 1) % n, (y + 1) % n, (z + 1) % n].add(weight_ * xw * yw * zw)
    #                                                                                                     zw)
    # # do the above with loop
    # for i in range(2):
    #     for j in range(2):
    #         for k in range(2):
    #             grid = grid.at[(x + i) % n, (y + j) % n, (z + k) % n].add(
    #                 weight_ * i * xw + (1 - i) * (1 - xw) * j * yw + (1 - i) * (1 - xw) * (1 - j) * (1 - yw) * k * zw)
    
    return field