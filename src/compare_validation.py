import jax.numpy as jnp
import pandas as pd
import sys
from config import load_config
import matplotlib.pyplot as plt
import os 
import nn
import jax

def main(argv) -> None:

    folder = "models/fno_layers"

    files = os.listdir(folder)
    files.sort()

    for i, file in enumerate(files):
        print(file)
        filename = os.path.join(folder, file)
        model, config, training_stats = nn.load(
            filename, nn.FNO, jax.nn.relu)
        
        validation_loss = training_stats['val_loss']
        times = training_stats['time']

        plt.plot(times, validation_loss, label=f"{i+1} layers")

    plt.xlabel("Compute (seconds)")
    plt.ylabel("MSE")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.savefig("img/validation.png")
    
if __name__ == "__main__":
    main(sys.argv[1:])