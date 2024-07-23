from .SpectralConvolution import SpectralConvolution
import jax
import jax.numpy as jnp

def test_spectral():

    key = jax.random.PRNGKey(42)

    k1, k2 = jax.random.split(key)

    # Create a random input tensor
    x = jax.random.uniform(k1, (16, 32, 32, 32))

    # Create a spectral convolution layer
    spectral_conv = SpectralConvolution(
        modes=4,
        n_channels=16, 
        activation=jnp.tanh,
        key=k2)

    # Apply the spectral convolution layer to the input tensor
    y = spectral_conv(x)

    # Check the output shape
    assert y.shape == (2, 16, 32, 32)