import jax.numpy as jnp
from .mass_assigment import linear_ma

def test_linear():

    pos = jnp.array([
        [0],
        [0.6125],
        [0]
    ])
    weight = jnp.array([
        1.0
    ])

    field = linear_ma(
        pos, weight, 4, 1
    )

    assert field[0, 2, 0] == 1.0