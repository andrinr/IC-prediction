import jax
import jax.numpy as jnp

def advance(x, _):
    x = x * 0.1
    return x, x

init = jax.random.uniform(jax.random.PRNGKey(0), (3, 1))
steps = 3
_, traj = jax.lax.scan(
    f = advance, init=init, xs=None, length=steps)

print(traj)