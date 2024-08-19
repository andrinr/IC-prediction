import jax.numpy as jnp
import jax
from .mass_assigment import linear_ma, cic_ma

def test_linear():

    pos = jnp.array([
        [0],
        [0.6125],
        [0]
    ])
    weight = jnp.array([1.0])

    field = linear_ma(
        pos, weight, 4, 1)

    assert field[0, 2, 0] == 1.0

def test_cic():
    N = 100
    pos = jax.random.uniform(jax.random.PRNGKey(0), (3, N))
    weight = jax.random.uniform(jax.random.PRNGKey(1), (N,))

    total = jnp.sum(weight)

    field = cic_ma(
        pos, weight, 4, 1)
    
    assert jnp.sum(field) == total
    
