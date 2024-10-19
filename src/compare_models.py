import sys
import os
# JAX 
import jax
import jax.numpy as jnp
# NVIDIA Dali
from nvidia.dali.plugin.jax import DALIGenericIterator
# Local
import nn
import data
import visualize
from config import Config
from cosmos import compute_overdensity, to_redshift, normalize, normalize_inv

def main(argv) -> None:

    folder = argv[0]

    # folder = "models/normalization"
    files = os.listdir(folder)
    files.sort()

    inputs = []
    predictions = []
    attributes_ls = []
    labels = []
    norm_functions = []

    # load config from first model, assuming data configs are all identical

    for i, file in enumerate(files):

        filename = os.path.join(folder, file)
        model, config, _ = nn.load_sequential_model(filename)

        # labels.append(config.normalizing_function)
        # labels.append(f"from z={to_redshift((config.file_index_start + config.file_index_stride) / 100):.1f}")
        # labels.append("+ potential" if config.include_potential else "")
        # labels.append(str(config.fno_hidden_channels) + " hidden channels")
        labels.append("mse" if i == 1 else "mse + power loss")
        # labels.append(str(config.fno_n_layers) + " layers")
        print((config.file_index_start + config.file_index_stride) / config.total_index_steps)

        norm_functions.append(config.normalizing_function)

            # Data Pipeline
        dataset = data.DirectorySequence(
            grid_size = config.input_grid_size,
            grid_directory = config.grid_dir,
            start = config.file_index_start,
            steps = config.file_index_steps,
            stride = config.file_index_stride,
            normalizing_function = config.normalizing_function,
            flip = config.flip,        
            type = "test")  

        data_pipeline = data.directory_sequence_pipe(dataset, config.grid_size)
        data_iterator = DALIGenericIterator(data_pipeline, ["data", "attributes"])

        sample = next(data_iterator)
        sequence = jax.device_put(sample['data'], jax.devices('gpu')[0])[1]
        attributes = jax.device_put(sample['attributes'], jax.devices('gpu')[0])[1]
        
        print(config.include_potential)
        pred_sequential = model(sequence, attributes, False, config.include_potential)

        predictions.append(pred_sequential)
        inputs.append(sequence)
        attributes_ls.append(attributes)

    visualize.compare(
        "img/compare.jpg", 
        sequences = inputs, 
        config = config,
        predictions = predictions,
        attributes = attributes_ls,
        labels = labels,
        norm_functions = norm_functions)

    # Delete Data Pipeline
    del data_pipeline

if __name__ == "__main__":
    main(sys.argv[1:])