import sys
# JAX 
import jax
import jax.numpy as jnp
# NVIDIA Dali
from nvidia.dali.plugin.jax import DALIGenericIterator
# Local
import nn
from data import VolumetricSequence, volumetric_sequence_pipe
import visualize
from config import Config

def main(argv) -> None:
   
    model_name = argv[0]

    model, config, training_stats = nn.load_sequential_model(
        model_name, jax.nn.relu)
    
    # Data Pipeline
    dataset = VolumetricSequence(
        grid_size = config.input_grid_size,
        directory = config.data_dir,
        start = config.start,
        steps = config.steps,
        stride = config.stride,
        flip=True,        
        type = "test")

    data_pipeline = volumetric_sequence_pipe(dataset, config.grid_size)
    data_iterator = DALIGenericIterator(data_pipeline, ["data", "steps", "means"])

    data = next(data_iterator)
    sequence = jax.device_put(data['data'], jax.devices('gpu')[0])[3]
    pred = model(sequence, False)
    pred_sequential = model(sequence, True)

    timeline = data["steps"][0]
    means = data["means"][0]

    visualize.sequence(
        "img/prediction_stepwise.jpg", 
        sequence = sequence, 
        config = config,
        sequence_prediction = pred,
        timeline = timeline,
        means = means)
    
    visualize.sequence(
        "img/prediction_sequential.jpg", 
        sequence = sequence, 
        config = config,
        sequence_prediction = pred_sequential,
        timeline = timeline,
        means = means)

    # Delete Data Pipeline
    del data_pipeline

if __name__ == "__main__":
    main(sys.argv[1:])