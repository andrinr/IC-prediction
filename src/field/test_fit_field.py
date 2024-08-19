from .fit_field import *
from .mass_assigment import *
import jax.numpy as jnp

def test_fit_field():
    key = jax.random.PRNGKey(0)
    key_field, key_opt = jax.random.split(key)
    N = 32
    field = jax.random.uniform(key_field, (N, N, N))
    print(f"field: {jnp.sum(field)}")   
    print(N**3*0.5)
    pos, mass = fit_field(
        key=key_opt,
        num_particles=N**3,
        field=field, 
        total_mass=N**3*0.5,
        iterations=1000,
        learning_rate=0.0001)
    
    field_pred = cic_ma(pos, mass, field.shape[0])

    assert jnp.allclose(field, field_pred, atol=1e-2)
    assert pos.shape == (3, 32)
    assert mass.shape == (32,)