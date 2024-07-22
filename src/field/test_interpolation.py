import jax.numpy as jnp
from .interpolation import lin_interp

def test_linear():

    field = jnp.array([
        [
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 0],
        ],
        [
            [0, 0, 0],
            [0, 100, 0],
            [0, 0, 0],
        ],
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 100],
        ],
    ])

    assert lin_interp([0, 0, 1/6],field) == 0.5
    assert lin_interp([0, 1/6, 1/6],field) == 0.75
    assert lin_interp([0, 1/2, 1/2],field) == 0.25