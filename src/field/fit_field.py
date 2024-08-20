import jax
import jax.numpy as jnp
import optax
from .mass_assigment import cic_ma
from typing import Tuple, NamedTuple

class Particles(NamedTuple):
    pos : jax.Array
    mass : jax.Array

def loss(
        pos : jax.Array,
        mass : jax.Array,
        field_truth : jax.Array):
    
    field_pred = cic_ma(
        pos,
        mass,
        field_truth.shape[0])
    
    return jnp.mean((field_pred - field_truth) ** 2)

def fit_field(
        key : jax.Array,
        num_particles : int,
        field : jax.Array,
        total_mass : float,
        iterations : float = 400,
        learning_rate : float = 0.005,
        ) -> Tuple[jax.Array, jax.Array]:

    # start random
    pos = jax.random.uniform(key, (3, num_particles))
    mass = jnp.ones(num_particles) * total_mass / num_particles

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(pos)

    grad_f = jax.grad(loss)

    @jax.jit
    def step(pos, opt_state):
        grad = grad_f(pos, mass, field)
        updates, opt_state = optimizer.update(grad, opt_state)
        pos = optax.apply_updates(pos, updates)
        return pos, opt_state
    
    for i in range(iterations):
        pos, opt_state = step(pos, opt_state)
        if i % 10 == 0:
            print(f"Loss: {loss(pos, mass, field)}")

    return pos, mass