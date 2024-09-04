import jax
import jax.numpy as jnp
from ..cosmos import Potential
import matplotlib.pyplot as plt

def compare(
        ouput_file : str,
        y_star : jax.Array,
        y : jax.Array,
        x : jax.Array):
    
    grid_size = y.shape[1]

    # transform to shape for matplotlib
    y_star = jnp.reshape(y_star[0], (grid_size, grid_size, grid_size, 1))
    y = jnp.reshape(y, (grid_size, grid_size, grid_size, 1))
    x = jnp.reshape(x[0], (grid_size, grid_size, grid_size, 1))

    min = jnp.min(jnp.array([y_star, y]))
    max = jnp.max(jnp.array([y_star, y]))

    power_spectrum = Potential(grid_size, 40)

    fig = plt.figure(figsize=(21, 14), layout="constrained")
    grid = fig.add_gridspec(nrows=2, ncols=3)

    ax_y_star = fig.add_subplot(grid[0, 0])
    ax_y = fig.add_subplot(grid[0, 1])
    ax_x = fig.add_subplot(grid[0, 2])
    ax_diff = fig.add_subplot(grid[1, 0])
    ax_power = fig.add_subplot(grid[1, 1:3])

    ax_y_star.axis('off')   
    ax_y_star.set_title("Y star")
    ax_y_star.imshow(y_star[grid_size // 2, : , :], vmin=min, vmax=max, cmap='inferno')
    k, power = power_spectrum(y_star[:, :, :, 0])
    ax_power.plot(k, power, label='y_star')

    ax_y.axis('off')
    ax_y.set_title("Y")
    ax_y.imshow(y[grid_size // 2, : , :], vmin=min, vmax=max, cmap='inferno')
    k, power = power_spectrum(y[:, :, :, 0])
    ax_power.plot(k, power, label='y')

    ax_x.axis('off')
    ax_x.set_title("X")
    ax_x.imshow(x[grid_size // 2, : , :], vmin=min, vmax=max, cmap='inferno')
    k, power = power_spectrum(x[:, :, :, 0])
    ax_power.plot(k, power, label='x')

    difference = y[grid_size // 2, : , :] - y_star[grid_size // 2, : , :]
    ax_diff.axis('off')
    ax_diff.set_title("Diff")
    ax_diff.imshow(difference, cmap='PiYG')

    ax_power.legend()
    ax_power.set_yscale('log')
    ax_power.set_xscale('log')

    plt.savefig(ouput_file)
