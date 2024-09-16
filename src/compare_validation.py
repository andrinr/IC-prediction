import jax.numpy as jnp
import pandas as pd
import sys
from config import load_config
import matplotlib.pyplot as plt
import os 
import nn
import jax

def main(argv) -> None:

    folder = "model/fno_layers"

    files = os.listdir(folder)
    files.sort()

    plots = []
    for file in files:
        model, config, training_stats = nn.load(
            file, nn.FNO, jax.nn.relu)
        
        
        


    data = pd.read_csv(config.train_log_file, header=0, index_col=0)

    print(data)

    plot = data.plot(title="Training and validation loss", ylabel="MSE", xlabel="epoch")
    fig = plot.get_figure()

    fig.savefig("img/training.png")


if __name__ == "__main__":
    main(sys.argv[1:])