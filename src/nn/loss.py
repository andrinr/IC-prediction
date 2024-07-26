import jax.numpy as jnp
import jax
import equinox as eqx
from functools import partial

@partial(jax.jit, static_argnums=1)
def mse_loss(params, static, x : jax.Array, y_star : jax.Array):
    model = eqx.combine(params, static)
    y = jax.vmap(model)(x)
    mse = jnp.sum((y_star - y) ** 2)
    print(mse.shape)
    return mse
