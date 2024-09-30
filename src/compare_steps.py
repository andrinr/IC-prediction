import jax.numpy as jnp
import numpy as np
import sys
import matplotlib.pyplot as plt
import os 
import nn
import jax
from matplotlib.cm import get_cmap
import matplotlib as mpl 

def main(argv) -> None:

    index_to_redshift = {
        0 : 49,
        3 : 13.122905,
        2 : 13.122905,
        5 : 7.11,
        10 : 4.251987,
        20 : 2.331247,
        30 : 2.331247,
        40 : 1.072709,
        50 : 0.7643640,
        60 : 0.538369,
        70 : 0.362698,
        80 : 0.220377,
        90 : 0.101529,
        100 : 0}

    folder = "models/ranges"

    files = os.listdir(folder)
    files.sort()

    configs = []
    losses = []
    red_starts = []

    for i, file in enumerate(files):
        filename = os.path.join(folder, file)
        model, config, training_stats = nn.load_sequential_model(
            filename, jax.nn.relu)
        
        parameter_count = nn.count_parameters(model)

        # m_params = int(jnp.log2(parameter_count))

        validation_loss = training_stats['stepwise_val_loss']

        configs.append(config)
        losses.append(jnp.array(validation_loss).max() - jnp.array(validation_loss).min())
        redshift = index_to_redshift[config.file_index_start]
        red_starts.append(redshift)

    red_starts = jnp.array(red_starts)

    bins = jnp.unique(red_starts)
    start_indices = jnp.digitize(red_starts, bins)

    plots_in_bin = jnp.zeros(bins.shape[0] + 1)

    losses = jnp.array(losses)
    min_loss = losses.min()
    max_loss = losses.max()

    cmap = get_cmap('viridis')

    height = 0.2

    for i in range(len(red_starts)):
        config = configs[i]
        loss = losses[i]
        start_index = start_indices[i]
        start_red = index_to_redshift[config.file_index_start]
        end_red = index_to_redshift[config.file_index_start + config.file_index_stride[0]]

        loss = (loss - min_loss) / max_loss

        print(loss)
        print(start_red)
        print(end_red)

        plt.barh(
            y = start_index + plots_in_bin[start_index] * height,
            width = end_red - start_red,
            left = start_red,
            height = height,
            color = cmap(loss**(1/2)),
            label=i)
        
        plots_in_bin = plots_in_bin.at[start_index].add(1)

    print(plots_in_bin)

    plt.gca().invert_xaxis()
    plt.title("Learning capacity of neural networks trained on different ranges")
    plt.xscale("log")
    plt.xlabel("Redshift")
    plt.ylabel("Starting Index")

    norm = mpl.colors.Normalize(vmin=0, vmax=1) 
    
    # creating ScalarMappable 
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm) 
    sm.set_array([]) 
    
    plt.colorbar(sm, ticks=np.linspace(0, 2, 8)) 

    plt.yticks([])
   

    plt.savefig("img/ranes.png")

    
if __name__ == "__main__":
    main(sys.argv[1:])