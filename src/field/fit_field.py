import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from functools import partial
from field import cic_ma
from typing import Tuple, NamedTuple

class Particles(NamedTuple):
    pos : jax.Array
    mass : jax.Array

def loss(
        particles : Particles,
        field_truth : jax.Array):
    
    field_pred = cic_ma(pos, mass, )

def fit_field(
        key : jax.Array,
        field : jax.Array, #[N, N, N]
        dx : float | jax.Array,
        total_mass : float,
        iterations : float = 100,
        learning_rate : float = 0.1,
        ) -> Tuple[jax.Array, jax.Array]:
    
    N = field.shape[0]
    # start random
    key_pos, key_mass = jax.random.split(key)
    pos = jax.random.uniform(key_pos, (3, N)) * N / 2 * dx
    mass = jax.random.uniform(key_mass) * total_mass / N

    particles = {
        "pos" : pos,
        "mass" : mass}
    
    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(particles)

    loss_grad = jax.grad(loss)

    @jax.jit
    def step(params, opt_state):
        grad = loss_grad(params, field, dx)
        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state




