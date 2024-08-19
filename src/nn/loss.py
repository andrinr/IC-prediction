import jax.numpy as jnp
import jax
import equinox as eqx
from functools import partial

@partial(jax.jit, static_argnums=1)
def mse_loss(params, static, x : jax.Array, y_star : jax.Array):
    # x, y_star shape : batch_size, 1, N, N, N
    model = eqx.combine(params, static)
    y = jax.vmap(model)(x)
    mse = jnp.mean((y_star - y) ** 2)
    return mse, y

@partial(jax.jit, static_argnums=1)
def spectral_loss(params, static, x : jax.Array, y_star : jax.Array):
    # x, y_star shape : batch_size, 1, N, N, N
    print(x.shape)
    N = x.shape[2]
    model = eqx.combine(params, static)
    y = jax.vmap(model)(x)
    mse = jnp.mean((y_star - y) ** 2)

    y_fs = jnp.fft.rfftn(x, s=(N, N, N), axes=(2, 3, 4))
    y_star_fs = jnp.fft.rfftn(y_star, s=(N, N, N), axes=(2, 3, 4))
    mse_fs = jnp.mean((jnp.abs(y_star_fs) - jnp.abs(y_fs.real)) ** 2)

    return mse_fs