import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

def sequence(
        ouput_file : str,
        sequence_prediction : jax.Array,
        sequence : jax.Array,
        timeline : jax.Array):
    
    frames = sequence.shape[0]
    grid_size = sequence.shape[2]

    # transform to shape for matplotlib
    sequence = jnp.reshape(
        sequence, (frames, grid_size, grid_size, grid_size, 1))
    sequence_prediction = jnp.reshape(
        sequence_prediction, (frames - 1, grid_size, grid_size, grid_size, 1))

    fig = plt.figure(figsize=(21, 14), layout="constrained")
    grid = fig.add_gridspec(nrows=2, ncols=frames)

    for frame in range(frames):
        
        min = jnp.min(sequence[frame])
        max = jnp.max(sequence[frame])
        if frame > 0:
            min = jnp.min(jnp.array([sequence_prediction[frame-1], sequence[frame]]))
            max = jnp.max(jnp.array([sequence_prediction[frame-1], sequence[frame]]))

        print(min)
        print(max)
        
        ax_seq = fig.add_subplot(grid[0, frame])
        ax_pred = fig.add_subplot(grid[1, frame])

        if frame > 0:
            ax_pred.axis('off')   
            ax_pred.set_title(f"pred t: {timeline[frame]}")
            ax_pred.imshow(sequence_prediction[frame - 1, grid_size // 2, : , :]) #, vmin=min, vmax=max, cmap='inferno')

        ax_seq.axis('off')   
        ax_seq.set_title(f"sim t: {timeline[frame]}")
        ax_seq.imshow(sequence[frame, grid_size // 2, : , :])#, vmin=min, vmax=max, cmap='inferno')

    plt.savefig(ouput_file)
