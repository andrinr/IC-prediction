import jax.numpy as jnp
import jax
import equinox as eqx
from functools import partial

@partial(jax.jit, static_argnums=1)
def mse_loss(params, static, x : jax.Array, y : jax.Array):
    model = eqx.combine(params, static)
    y_pred = jax.vmap(model)(x)
    mse = jnp.mean(jnp.square(y_pred - y))
    return mse
