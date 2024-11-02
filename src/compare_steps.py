import jax.numpy as jnp
import numpy as np
import sys
import matplotlib.pyplot as plt
import os
import nn
import jax
from matplotlib.cm import get_cmap
import matplotlib as mpl
from cosmos import to_redshift

def main(argv) -> None:

    folder = "/data/arehma/models/ranges-fd"
    files = os.listdir(folder)

    configs = []
    losses = []
    red_starts = []

    for i, file in enumerate(files):
        filename = os.path.join(folder, file)
        model, config, training_stats = nn.load_sequential_model(filename)
        
        parameter_count = nn.count_parameters(model)
        
        validation_loss = training_stats['metric_step']['val_RSE']
        print(filename)
        print(validation_loss)
        configs.append(config)
        losses.append(jnp.array(validation_loss).min())
        redshift = to_redshift(config.file_index_start / 100)
        red_starts.append(redshift)

    combined = list(zip(losses, configs, red_starts))
    sorted_combined = sorted(combined, key=lambda x: x[0])

    losses, configs, red_starts = zip(*sorted_combined)
    red_starts = jnp.array(red_starts)
    bins = jnp.unique(red_starts)
    start_indices = jnp.digitize(red_starts, bins)
    plots_in_bin = jnp.zeros(bins.shape[0] + 1)
    losses = jnp.array(losses)
    min_loss = losses.min()
    max_loss = losses.max()
    cmap = get_cmap('PiYG').reversed()
    height = 0.12

    fig, ax = plt.subplots()  # Create subplots and get the axis
    for i in range(len(red_starts)):
        config = configs[i]
        loss = losses[i]
        start_index = start_indices[i]
        start_red = to_redshift(config.file_index_start / 100)
        end_red = to_redshift((config.file_index_start + config.file_index_stride[0])/100)
        norm_loss = (loss - min_loss) / (max_loss - min_loss)  # Normalize loss
        color = cmap(norm_loss)
        
        y = start_index + plots_in_bin[start_index] * height
        x_start = 1 / (1 + start_red)
        x_end = 1 / (1 + end_red)
        
        ax.barh(
            y=y,
            width=x_end - x_start,
            left=x_start,
            height=height,
            color=color,
            label=i,
            edgecolor = "black",
            linewidth=1
        )
        
        plots_in_bin = plots_in_bin.at[start_index].add(1)


    ax.set_title("RSE of models trained of different ranges")
    ax.set_xlabel("Expansion Factor")
    ax.set_ylabel("Predicted Expansion Factor")
    ax.set_yticks([])
    ax.grid(True)
    ax.xaxis.set_inverted(True)

    # Create a colorbar
    norm = mpl.colors.Normalize(vmin=min_loss, vmax=max_loss)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # dummy array for ScalarMappable
    
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', ticks=np.linspace(min_loss, max_loss, 8))
    cbar.set_label('Relative Squared Error')

    plt.savefig("img/ranges.png")
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])