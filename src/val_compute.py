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
    if param == "normalization":
        folder = "models/normalization"

    files = os.listdir(folder)
    files.sort()

    for i, file in enumerate(files):
        print(file)
        filename = os.path.join(folder, file)
        model, config, training_stats = nn.load_sequential_model(
            filename)
        
        parameter_count = nn.count_parameters(model)
        print(f'Number of parameters: {parameter_count}')

        m_params = int(jnp.log2(parameter_count))

        validation_loss = jnp.array(training_stats['metric_step']['val_mse'])[2:]
        training_loss = jnp.array(training_stats['metric_step']['train_mse'])[2:]
        print(validation_loss)
        n = len(validation_loss)
        if param == "stride" or param == "normalization":
            times = jnp.linspace(0, n-1,  n)
            validation_loss = abs(validation_loss - validation_loss[0]) / validation_loss.max()
            training_loss = abs(training_loss - training_loss[0]) / training_loss.max()
            print(times)
        else:
            times = training_stats['time']

        if param == "layers":
            plt.plot(times, validation_loss, label=f"{2**i} layers; 2^{m_params} params")
        if param == "channels":
            plt.plot(times, validation_loss, label=f"{2**i} channels; 2^{m_params} params")
        if param == "stride":
            plt.plot(times, validation_loss, label=f"{2**(2+i)} stride; 2^{m_params} params")
        if param == "normalization":
            plt.plot(times, validation_loss, label=f"val {config.normalizing_function}")
            # plt.plot(times, training_loss, label=f"train {config.normalizing_function}")

        
    if param == "stride" or param == "normalization":
        plt.xlabel("Epoch")
    elif param == "ranges":
        plt.gca().invert_xaxis()
        plt.xscale("log")
        plt.xlabel("Redshifts")
    else:
        plt.xlabel("Compute (seconds)")
    
    plt.ylabel("Absolute Validation Loss Decrease")

    if param == "layers" or param == "channels" or param == "normalization":
        plt.yscale("log")
        # plt.xscale("log")
    plt.legend()

    if param == "layers":
        plt.savefig("img/layer_compute.png")
    if param == "channels":
        plt.savefig("img/channels_compute.png")
    if param == "stride":
        plt.savefig("img/stride.png")
    if param == "normalization":
        plt.savefig("img/normalization.png")

    

if __name__ == "__main__":
    main(sys.argv[1:])