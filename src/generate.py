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
from config import Config
import field

def main(argv) -> None:
   
    model_name = argv[0]

    model, config, training_stats = nn.load(
        model_name, nn.SequentialFNO, jax.nn.relu)
    
    config = Config(**config)
    
    # Data Pipeline
    dataset = VolumetricSequence(
        grid_size = config.input_grid_size,
        directory = config.data_dir,
        start = 0,
        steps = config.steps,
        stride = config.stride,
        flip=True,        
        type = "test")

    data_pipeline = volumetric_sequence_pipe(dataset, config.grid_size)
    data_iterator = DALIGenericIterator(data_pipeline, ["data", "steps", "means"])

    data = next(data_iterator)
    sequence = jax.device_put(data['data'], jax.devices('gpu')[0])[0]
    # pred = model(sequence, False)
    # pred_sequential = model(sequence, True)

    timeline = data["steps"][0]
    means = data["means"][0]

    gpus = jax.devices("gpu")
    means = jax.device_put(means, gpus[0])

    rho = cosmos.compute_rho(sequence[-1], means[-1])

    total_mass = jnp.sum(rho)

    lagrangian_position, euelerian_position, mass = field.fit_field(
        jax.random.PRNGKey(0),
        config.num_particles,
        rho,
        total_mass,
        400)
    
    a = 1 / (1 + config.redshift_start)
    
    #euelerian_position = lagrangian_position + dspls * m_Dplus;
    D_plus = cosmos.growth_factor_approx(a, config.Omega_M, config.Omega_L)
    
    displacement = (euelerian_position - lagrangian_position) / D_plus

    D_plus_da = cosmos.growth_factor_approx_deriv(config.Omega_M, config.Omega_L)

    velocity = displacement * D_plus_da

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