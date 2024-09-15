import sys
# JAX 
import jax
import jax.numpy as jnp
# NVIDIA Dali
from nvidia.dali.plugin.jax import DALIGenericIterator
# Local
import nn
from data import VolumetricSequence, volumetric_sequence_pipe, generate_tipsy
import cosmos
from config import load_config
import field

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
    delta_pred = jnp.zeros_like(sequence)

    n_frames = sequence.shape[0]
    delta_pred = delta_pred.at[0].set(sequence[0])
    for i in range(1, n_frames):
        print(i)
        delta_pred = delta_pred.at[i].set(model(delta_pred[i-1]))

    time = data["steps"][0]
    mean = data["means"][0]

    rho = cosmos.compute_density(delta_pred, mean)

    total_mass = jnp.sum(rho)

    lagrangian_position, euelerian_position, mass = field.fit_field(
        jax.random.PRNGKey(0),
        config.num_particles,
        rho,
        total_mass,
        400)
    
    #euelerian_position = lagrangian_position + dspls * m_Dplus;
    D_plus = cosmos.compute_growth_factor(time, config.Omega_M, config.Omega_L)
    
    displacement = (euelerian_position - lagrangian_position) / D_plus

    v_fact = cosmos.compute_v_factor(time, config.Omega_M, config.Omega_L)

    velocity = displacement * D_plus * v_fact

    generate_tipsy(
        config.output_tipsy_file,
        lagrangian_position,
        velocity,
        mass,
        config.box_size,
        config.dt_PKDGRAV3)
    
    # Delete Data Pipeline
    del data_pipeline

if __name__ == "__main__":
    main(sys.argv[1:])