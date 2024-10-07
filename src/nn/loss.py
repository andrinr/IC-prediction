import jax.numpy as jnp
import jax
import equinox as eqx
from functools import partial
from cosmos import PowerSpectrum, compute_overdensity, normalize_inv

@partial(jax.jit)
def mse(prediction : jax.Array, truth : jax.Array):
    return jnp.mean((prediction - truth) ** 2)

def power_loss(    
        prediction : jax.Array, 
        truth : jax.Array,
        attributes : jax.Array):
    
    #truth / prediction: [Batch, Frames, Channels, Depth, Height, Width]
    #attributes: [Batch, Frames, normaliztion_attributes]

    normalize_inv_func = lambda x, y : normalize_inv(x, y, "log_growth")

    b, f, c, d, w, h = prediction.shape
    power_spectrum = PowerSpectrum(
        d, 20)

    normalize_map = jax.vmap(jax.vmap(normalize_inv_func))
    power_map = jax.vmap(jax.vmap(power_spectrum))

    rho_truth = normalize_map(truth, attributes)
    rho_pred = normalize_map(prediction, attributes)

    # mass_loss = mass_conservation_loss(rho_pred, rho_truth)
    k, p_truth = power_map(rho_truth[:, :, 0])
    k, p_pred = power_map(rho_pred[:, :, 0])

    return mse(jnp.log10(p_truth), jnp.log10(p_pred))

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