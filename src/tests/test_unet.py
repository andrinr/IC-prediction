import jax
import jax.numpy as jnp
from nn import UNet

def test_unet():

    key = jax.random.PRNGKey(42)

    k1, k2 = jax.random.split(key)

    # Create a random input tensor
    x = jax.random.uniform(k1, (1, 32, 32, 32))

    # Create a spectral convolution layer
    model = UNet(
        num_spatial_dims=3,
        in_channels=1,
        out_channels=1,
        hidden_channels=8,
        num_levels=1,
        activation=jax.nn.relu,
        padding='SAME',
        padding_mode='CIRCULAR',	
        key=k1)

    # Apply the spectral convolution layer to the input tensor
    y = model(x)

    # Check the output shape
    assert y.shape == (1, 32, 32, 32)