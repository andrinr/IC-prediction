import jax.numpy as jnp
import jax
import equinox as eqx
from functools import partial

@partial(jax.jit, static_argnums=1)
def mse_loss(params, static, x : jax.Array, y_star : jax.Array):
    # x, y_star shape : batch_size, 1, N, N, N
    model = eqx.combine(params, static)
    y = jax.vmap(model)(x)
    mse = jnp.sum((y_star - y) ** 2)
    return mse

@partial(jax.jit, static_argnums=1)
def furier_loss(params, static, x : jax.Array, y_star : jax.Array):
    # x, y_star shape : batch_size, 1, N, N, N
    N = x.shape[2]
    model = eqx.combine(params, static)
    y = jax.vmap(model)(x)
    y_fs = jnp.fft.rfftn(x, s=(N, N, N), axes=(2, 3, 4))
    y_star_fs = jnp.fft.rfftn(y_star, s=(N, N, N), axes=(2, 3, 4))
    mse = jnp.sum((y_star - y) ** 2)
    mse_fs = jnp.sum((y_star_fs - y_fs) ** 2)
    return mse + mse_fs
