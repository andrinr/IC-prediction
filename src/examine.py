import sys
# JAX 
import jax
import jax.numpy as jnp
# NVIDIA Dali
from nvidia.dali.plugin.jax import DALIGenericIterator
# Local
import nn
import data
import visualize
from config import Config, load_config

def main(argv) -> None:
   
    config = load_config(argv[0])

    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_disable_jit", False)

    dataset_params = {
        "grid_size" : config.input_grid_size,
        "grid_directory" : config.grid_dir,
        "start" : config.file_index_start,
        "steps" : config.file_index_steps,
        "stride" : config.file_index_stride,
        "normalizing_function" : config.normalizing_function,
        "flip" : config.flip,
        "type" : "test"}
    
    dataset = data.DirectorySequence(**dataset_params)
    data_pipeline = data.directory_sequence_pipe(dataset, config.grid_size)
    data_iterator = DALIGenericIterator(data_pipeline, ["data", "attributes"])

    sample = next(data_iterator)
    sequence = jax.device_put(sample['data'], jax.devices('gpu')[0])[0]

    attributes = sample["attributes"][0]

    visualize.compare(
        "img/data_distr.jpg", 
        sequence_curr = sequence, 
        sequence_prediction=None,
        config = config,
        attributes = attributes)
    
    # Delete Data Pipeline
    del data_pipeline

if __name__ == "__main__":
    main(sys.argv[1:])