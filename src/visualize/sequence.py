import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from config import Config
from cosmos import PowerSpectrum

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

    fig = plt.figure(figsize=(10, 6), layout="constrained")
    grid = fig.add_gridspec(nrows=3, ncols=frames)

    power_spectrum = PowerSpectrum(grid_size, 30)

    ax_power = fig.add_subplot(grid[2, :])

    for frame in range(frames):

        mean = means[frame]
        mean = jax.device_put(mean, device=jax.devices("gpu")[0])
        print(mean)

        k, power = power_spectrum(sequence[frame, :, :, :, 0])
        ax_power.plot(k, power, label=f"sim t: {timeline[frame]}")

        if frame < frames - 1:
            k, power = power_spectrum(sequence_prediction[frame, :, :, :, 0])
            ax_power.plot(k, power, label=f"pred t: {timeline[frame]}")
        
        min = jnp.min(sequence[frame, grid_size // 2])
        max = jnp.max(sequence[frame, grid_size // 2])
        if frame < frames - 1:
            min = jnp.min(jnp.array([sequence_prediction[frame, grid_size // 2], sequence[frame, grid_size // 2]]))
            max = jnp.max(jnp.array([sequence_prediction[frame, grid_size // 2], sequence[frame, grid_size // 2]]))

        ax_seq = fig.add_subplot(grid[0, frame])

        if frame > 0:
            ax_pred = fig.add_subplot(grid[1, frame])
            ax_pred.axis('off')   
            ax_pred.set_title(f"pred t: {timeline[frame]}")
            ax_pred.imshow(sequence_prediction[frame, grid_size // 2, : , :], cmap='inferno') #vmin=min, vmax=max)

        ax_seq.axis('off')   
        ax_seq.set_title(f"sim t: {timeline[frame]}")
        ax_seq.imshow(sequence[frame, grid_size // 2, : , :], cmap='inferno')
                    #   vmin=jnp.percentile(sequence[frame, grid_size // 2, : , :], 10),
                    # vmax = jnp.percentile(sequence[frame, grid_size // 2, : , :], 90))


    # ax_power.set_yscale('log')
    # ax_power.set_xscale('log')
    # ax_power.legend()

    # ax_power.set_xticks(jnp.linspace(0, config.box_size, 5))
    # ax_power.set_yticks(jnp.linspace(0, 0.1, 5))

    plt.savefig(ouput_file)
