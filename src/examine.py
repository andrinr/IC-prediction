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
        "tipsy_directory" : config.tipsy_dir,
        "start" : config.start,
        "steps" : config.steps,
        "stride" : config.stride,
        "flip" : True,
        "type" : "test"}
    
    dataset = data.DirectorySequence(**dataset_params)
    data_pipeline = data.directory_sequence_pipe(dataset, config.grid_size)
    data_iterator = DALIGenericIterator(data_pipeline, ["data", "step", "attributes"])

    sample = next(data_iterator)
    sequence = jax.device_put(sample['data'], jax.devices('gpu')[0])[0]
   
    timeline = sample["step"][0]
    attributes = sample["attributes"][0]

    visualize.sequence_examine(
        "img/data_distr.jpg", 
        sequence = sequence, 
        config = config,
        timeline = timeline,
        attributes = attributes)
    
    # Delete Data Pipeline
    del data_pipeline

if __name__ == "__main__":
    main(sys.argv[1:])