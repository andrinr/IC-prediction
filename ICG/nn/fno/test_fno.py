from .spectral_convolution import SpectralConvolution
from .furier_layer import FurierLayer
from .fno import FNO
import jax
import jax.numpy as jnp

def test_fno():

    key = jax.random.PRNGKey(42)

    k1, k2 = jax.random.split(key)

    # Create a random input tensor
    x = jax.random.uniform(k1, (16, 32, 32, 32))

    # Create a spectral convolution layer
    spectral_conv = SpectralConvolution(
        modes=6,
        n_channels=16, 
        key=k2)

    # Apply the spectral convolution layer to the input tensor
    y = spectral_conv(x)

    # Check the output shape
    assert y.shape == (16, 32, 32, 32)

    furier_layer = FurierLayer(
        modes=6,
        n_channels=16,
        activation=jnp.tanh,
        key=k2)
    
    y = furier_layer(x)

    assert y.shape == (16, 32, 32, 32)

    # Create a random input tensor
    x = jax.random.uniform(k1, (1, 32, 32, 32))

    fno = FNO(
        modes=6,
        hidden_channels=16,
        activation=jnp.tanh,
        n_furier_layers=2,
        key=k2)
    
    y = fno(x)

    assert y.shape == (1, 32, 32, 32)
