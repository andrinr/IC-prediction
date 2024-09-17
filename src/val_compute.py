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

    files = os.listdir(folder)
    files.sort()

    for i, file in enumerate(files):
        print(file)
        filename = os.path.join(folder, file)
        model, config, training_stats = nn.load(
            filename, nn.FNO, jax.nn.relu)
        
        parameter_count = nn.count_parameters(model)
        print(f'Number of parameters: {parameter_count}')

        m_params = int(jnp.log2(parameter_count))
        
        validation_loss = training_stats['val_loss']
        times = training_stats['time']

        if param == "layers":
            plt.plot(times, validation_loss, label=f"{2**i} layers; 2^{m_params} params")
        if param == "channels":
            plt.plot(times, validation_loss, label=f"{2**i} channels; 2^{m_params} params")
        
    plt.xlabel("Compute (seconds)")
    plt.ylabel("Validation Loss")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()

    if param == "layers":
        plt.savefig("img/layer_compute.png")
    if param == "channels":
        plt.savefig("img/channels_compute.png")

if __name__ == "__main__":
    main(sys.argv[1:])