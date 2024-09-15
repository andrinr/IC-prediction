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
from config import load_config

def main(argv) -> None:
   
    config = load_config(argv[0])

    # Data Pipeline
    dataset = VolumetricSequence(
        grid_size = config.input_grid_size,
        directory = config.data_dir,
        start = 0,
        steps = config.stride,
        stride = config.stride,
        flip=True)

    data_pipeline = volumetric_sequence_pipe(dataset, config.grid_size)
    data_iterator = DALIGenericIterator(data_pipeline, ["sequence", "steps", "means"])

    model = nn.load(
        config.model_params_file, nn.FNO, jax.nn.relu)

    data = next(data_iterator)
    sequence = jax.device_put(data['sequence'], jax.devices('gpu')[0])[0]
    pred = jnp.zeros_like(sequence)

    n_frames = sequence.shape[0]
    pred = pred.at[0].set(sequence[0])
    for i in range(1, n_frames):
        print(i)
        pred = pred.at[i].set(model(pred[i-1]))

    timeline = data["steps"][0]
    means = data["means"][0]

    visualize.sequence(
        "img/seq.jpg", 
        sequence = sequence, 
        config = config,
        sequence_prediction = pred,
        timeline = timeline,
        means = means)

    # Delete Data Pipeline
    del data_pipeline

if __name__ == "__main__":
    main(sys.argv[1:])