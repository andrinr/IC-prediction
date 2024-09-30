import jax.numpy as jnp
import pandas as pd
import sys
from config import load_config
import matplotlib.pyplot as plt
import os 
import nn
import jax

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
        losses.append(validation_loss)
        redshift = index_to_redshift[config.file_index_start]
        red_starts.append(redshift)

    red_starts = jnp.array(red_starts)

    bins = jnp.unique(red_starts)
    start_indices = jnp.digitize(red_starts, bins)

    plots_in_bin = jnp.zeros(bins.shape[0] + 1)

    width = 0.1

    for i in range(len(red_starts)):
        config = configs[i]
        loss = losses[i]
        start_index = start_indices[i]
        start_red = index_to_redshift[config.file_index_start]
        end_red = index_to_redshift[config.file_index_start + config.file_index_stride[0]]

        plt.barh(
            y = start_index + plots_in_bin[start_index] * width,
            width = end_red - start_red,
            left = start_red,
            height = width,
            label=i)
        
        plots_in_bin = plots_in_bin.at[start_index].add(1)

    print(plots_in_bin)

    plt.gca().invert_xaxis()
    plt.xscale("log")
    plt.xlabel("Redshift")
    plt.ylabel("Starting Index")
   
    plt.ylabel("Validation Loss")

    plt.savefig("img/ranes.png")

    
if __name__ == "__main__":
    main(sys.argv[1:])