import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from ..config import Config

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
        sequence_prediction, (frames, grid_size, grid_size, grid_size, 1))

    fig = plt.figure(figsize=(10, 6), layout="constrained")
    grid = fig.add_gridspec(nrows=2, ncols=frames)

    for frame in range(frames):

        # TODO: add colorbar, vscale
        
        min = jnp.min(sequence[frame, grid_size // 2])
        max = jnp.max(sequence[frame, grid_size // 2])
        if frame > 0:
            min = jnp.min(jnp.array([sequence_prediction[frame, grid_size // 2], sequence[frame, grid_size // 2]]))
            max = jnp.max(jnp.array([sequence_prediction[frame, grid_size // 2], sequence[frame, grid_size // 2]]))

        ax_seq = fig.add_subplot(grid[0, frame])
        ax_pred = fig.add_subplot(grid[1, frame])

        if frame > 0:
            ax_pred.axis('off')   
            ax_pred.set_title(f"pred t: {timeline[frame]}")
            ax_pred.imshow(sequence_prediction[frame, grid_size // 2, : , :], vmin=min, vmax=max, cmap='inferno')

        ax_seq.axis('off')   
        ax_seq.set_title(f"sim t: {timeline[frame]}")
        ax_seq.imshow(sequence[frame, grid_size // 2, : , :])#, vmin=min, vmax=max, cmap='inferno')

    plt.savefig(ouput_file)
