import jax.numpy as jnp
import jax
from .interpolation import linear_interp

def test_linear():

    field = jnp.array([
        [
            [0, 1],
            [0, 1]
        ],
        [
            [0, 0],
            [0, 0]
        ]
    ])

    x = [0.1, 0, 0]

    value = linear_interp(
        x,
        field, 
        1)
    
    print(value)
    
    assert False

