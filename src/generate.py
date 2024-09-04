import sys
# JAX 
import jax
import jax.numpy as jnp
# NVIDIA Dali
from nvidia.dali.plugin.jax import DALIGenericIterator
# Local
import nn
from data import VolumetricSequence, volumetric_sequence_pipe, generate_tipsy
import visualize
import cosmos
from config import load_config
import field

def main(argv) -> None:
   
    config = load_config(argv[0])

    # Data Pipeline
    dataset = VolumetricSequence(
        grid_size = config.input_grid_size,
        directory = config.train_data_dir,
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

    time = data["steps"][0]
    mean = data["means"][0]

    potential = cosmos.Potential(config.grid_size)(pred)
    
    velocity_field = cosmos.compute_velocity(potential, config.dt_PKDGRAV3)

    rho = cosmos.compute_density(pred, mean)

    total_mass = jnp.sum(rho)

    position, mass = field.fit_field(
        jax.random.PRNGKey(0),
        config.num_particles,
        rho,
        total_mass,
        400)
    
    velocity = field.bilinear_interp(position.pos, velocity_field)

    generate_tipsy(
        config.output_tipsy_file,
        position,
        velocity,
        mass,
        config.redshift_start)

    # Delete Data Pipeline
    del data_pipeline

if __name__ == "__main__":
    main(sys.argv[1:])