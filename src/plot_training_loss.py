import jax.numpy as jnp
import pandas as pd
import sys
from config import load_config
import matplotlib.pyplot as plt

def main(argv) -> None:
   
    config = load_config(argv[0])

    data = pd.read_csv(config.train_log_file, header=0, index_col=0)

    print(data)

    plot = data.plot(title="Training and validation loss", ylabel="MSE", xlabel="epoch")
    fig = plot.get_figure()

    fig.savefig("img/training.png")


if __name__ == "__main__":
    main(sys.argv[1:])