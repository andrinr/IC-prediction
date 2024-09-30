import jax.numpy as jnp
import pandas as pd
import sys
from config import load_config
import matplotlib.pyplot as plt
import os 
import nn
import jax

def main(argv) -> None:

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
        red_starts.append(config.redshifts[0])

    red_starts = jnp.array(red_starts)

    bins = jnp.unique(red_starts)
    start_indices = jnp.digitize(red_starts, bins)

    for i in range(len(red_starts)):
        config = configs[i]
        loss = losses[i]
        start_index = start_indices[i]

        plt.bar(
            height = abs(config.redshifts[1]-config.redshifts[1]), 
            bottom = config.redshifts[0],
            [start_index, start_index], label=f"start {config.redshifts[0]:1f}, end {config.redshifts[1]:1f}")

    plt.gca().invert_xaxis()
    plt.xscale("log")
    plt.xlabel("Redshifts")
   
    plt.ylabel("Validation Loss")

    plt.savefig("img/ranes.png")

    
if __name__ == "__main__":
    main(sys.argv[1:])