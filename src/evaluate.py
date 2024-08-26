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
from config import load_config

def main(argv) -> None:
   
    config = load_config(argv[0])

    # Data Pipeline
    dataset = data.VolumetricSequence(
        grid_size = config.input_grid_size,
        directory = config.train_data_dir,
        start = 0,
        steps = config.stride,
        stride = config.stride,
        flip=True)

    data_pipeline = data.volumetric_sequence_pipe(dataset, config.grid_size)
    data_iterator = DALIGenericIterator(data_pipeline, ["sequence", "time"])

    model = nn.load(
        config.model_params_file, "sq_fno.eqx", nn.FNO, jax.nn.relu)

    data = next(data_iterator)
    sequence = jax.device_put(data['sequence'], jax.devices('gpu')[0])[0]
    pred = jnp.zeros_like(sequence)

    n_frames = sequence.shape[0]
    pred = pred.at[0].set(sequence[0])
    for i in range(1, n_frames):
        print(i)
        pred = pred.at[i].set(model(pred[i-1]))

    timeline = data["time"][0]

    visualize.sequence(
        "seq.jpg", 
        sequence = sequence, 
        sequence_prediction = pred,
        timeline = timeline)

    # Delete Data Pipeline
    del data_pipeline

if __name__ == "__main__":
    main(sys.argv[1:])