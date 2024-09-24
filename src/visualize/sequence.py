import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from config import Config
from cosmos import PowerSpectrum
from mpl_toolkits.axes_grid1 import make_axes_locatable

def sequence(
        ouput_file : str,
        sequence_prediction : jax.Array,
        config : Config,
        sequence : jax.Array,
        timeline : jax.Array,
        means : jax.Array):
    
    frames = sequence.shape[0]
    grid_size = sequence.shape[2]

    # transform to shape for matplotlib
    sequence = jnp.reshape(
        sequence, (frames, grid_size, grid_size, grid_size, 1))
    sequence_prediction = jnp.reshape(
        sequence_prediction, (frames-1, grid_size, grid_size, grid_size, 1))

    fig = plt.figure(figsize=(8, 6), layout="constrained")
    grid = fig.subplots(nrows=2, ncols=frames)

    power_spectrum = PowerSpectrum(grid_size, 30)

    ax_power = grid[1, 0]

    for frame in range(frames):

        mean = means[frame]
        mean = jax.device_put(mean, device=jax.devices("gpu")[0])

        k, power = power_spectrum(sequence[frame, :, :, :, 0])
        ax_power.plot(k, power, label=f"sim t: {timeline[frame]}")

        # if frame < frames - 1:
        #     k, power = power_spectrum(sequence_prediction[frame, :, :, :, 0])
        #     ax_power.plot(k, power, label=f"pred t: {timeline[frame]}")
        
        if frame > 0:
            ax_pred = grid[1, frame]
            ax_pred.set_title(f"pred t: {timeline[frame]}")
            im_pred = ax_pred.imshow(sequence_prediction[frame, grid_size // 2, : , :], cmap='inferno')
            fig.colorbar(im_pred, ax=ax_pred, orientation='vertical', location='right')

            k, power = power_spectrum(sequence_prediction[frame, :, :, :, 0])
            ax_power.plot(k, power, label=f"pred t: {timeline[frame]}")

        ax_seq = grid[0, frame]
        ax_seq.set_title(f"sim t: {timeline[frame]}")
        im_seq = ax_seq.imshow(sequence[frame, grid_size // 2, : , :], cmap='inferno')
        fig.colorbar(im_seq, ax=ax_seq, orientation='vertical', location='right')

    ax_power.set_yscale('log')
    ax_power.set_xscale('log')
    ax_power.legend()
    

    plt.savefig(ouput_file)
