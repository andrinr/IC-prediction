import jax.numpy as jnp
import pandas as pd
import sys
from config import load_config
import matplotlib.pyplot as plt
import os 
import nn
import jax

def main(argv) -> None:

    param = argv[0]

    if param == "layers":
        folder = "models/fno_layers"
    if param == "channels":
        folder = "models/fno_channels"
    if param == "stride":
        folder = "models/stride"
    if param == "ranges":
        folder = "models/ranges"

    files = os.listdir(folder)
    files.sort()

    for i, file in enumerate(files):
        print(file)
        filename = os.path.join(folder, file)
        model, config, training_stats = nn.load_sequential_model(
            filename, jax.nn.relu)
        
        parameter_count = nn.count_parameters(model)
        print(f'Number of parameters: {parameter_count}')

        m_params = int(jnp.log2(parameter_count))

        validation_loss = training_stats['stepwise_val_loss']
        n = len(training_stats['stepwise_val_loss'])
        if param == "stride" or param == "ranges":
            times = jnp.linspace(0, n-1,  n)
            print(times)
        else:
            times = training_stats['time']

        if param == "layers":
            plt.plot(times, validation_loss, label=f"{2**i} layers; 2^{m_params} params")
        if param == "channels":
            plt.plot(times, validation_loss, label=f"{2**i} channels; 2^{m_params} params")
        if param == "stride":
            plt.plot(times, validation_loss, label=f"{2**(2+i)} stride; 2^{m_params} params")
        if param == "ranges":
            plt.plot([config.redshifts[0], config.redshifts[1]], [i, i], label=f"start {config.redshifts[0]:1f}, end {config.redshifts[1]:1f}")
        
    if param == "stride":
        plt.xlabel("Epoch")
    elif param == "ranges":
        plt.gca().invert_xaxis()
        plt.xscale("log")
        plt.xlabel("Redshifts")
    else:
        plt.xlabel("Compute (seconds)")
    
    plt.ylabel("Validation Loss")

    if param == "layers" or param == "channels":
        plt.yscale("log")
        plt.xscale("log")
    # plt.legend()

    if param == "layers":
        plt.savefig("img/layer_compute.png")
    if param == "channels":
        plt.savefig("img/channels_compute.png")
    if param == "stride":
        plt.savefig("img/stride.png")
    if param == "ranges":
        plt.savefig("img/ranes.png")

    

if __name__ == "__main__":
    main(sys.argv[1:])