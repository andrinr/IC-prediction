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
from config import Config

def main(argv) -> None:
   
    model_name = argv[0]

    model, config, training_stats = nn.load_sequential_model(
        model_name, jax.nn.relu)
    
    # Data Pipeline
    dataset = data.DirectorySequence(
        grid_size = config.input_grid_size,
        grid_directory = config.grid_dir,
        tipsy_directory = config.tipsy_dir,
        start = config.start,
        steps = config.steps,
        stride = config.stride,
        flip=True,        
        type = "test")

    data_pipeline = data.directory_sequence_pipe(dataset, config.grid_size)
    data_iterator = DALIGenericIterator(data_pipeline, ["data", "step", "attributes"])

    # dummy_val_dataset = data.DummyData(
    #     batch_size=10,
    #     steps = config.steps,
    #     grid_size=config.input_grid_size)
    # data_pipeline = data.dummy_sequence_pipe(dummy_val_dataset, config.grid_size)
    # data_iterator = DALIGenericIterator(data_pipeline, ["data", "step", "mean"])

    sample = next(data_iterator)
    sequence = jax.device_put(sample['data'], jax.devices('gpu')[0])[1]
    pred = model(sequence, False)
    pred_sequential = model(sequence, True)

    timeline = sample["step"][0]
    attributes = sample["attributes"][0]

    visualize.sequence(
        "img/prediction_stepwise.jpg", 
        sequence = sequence, 
        config = config,
        sequence_prediction = pred,
        timeline = timeline,
        attributes = attributes)

    visualize.sequence(
        "img/prediction_sequential.jpg", 
        sequence = sequence, 
        config = config,
        sequence_prediction = pred_sequential,
        timeline = timeline,
        attributes = attributes)
    
    # visualize.sequence(
    #     "img/prediction_sequential.jpg", 
    #     sequence = sequence, 
    #     config = config,
    #     sequence_prediction = pred_sequential,
    #     timeline = timeline,
    #     attributes = attributes)

    # Delete Data Pipeline
    del data_pipeline

if __name__ == "__main__":
    main(sys.argv[1:])