import sys
# JAX 
import jax
import jax.numpy as jnp
# NVIDIA Dali
from nvidia.dali.plugin.jax import DALIGenericIterator
# Local
import nn
from data import DirectorySequence, directory_sequence_pipe, generate_tipsy
import cosmos
from config import Config
import field

def main(argv) -> None:
   
    model_name = argv[0]

    model, config, training_stats = nn.load_sequential_model(
        model_name, jax.nn.relu)
    
    # Data Pipeline
    dataset = DirectorySequence(
        grid_size = config.input_grid_size,
        directory = config.data_dir,
        start = config.start,
        steps = config.steps,
        stride = config.stride,
        flip=True,        
        type = "test")


    data_pipeline = directory_sequence_pipe(dataset, config.grid_size)
    data_iterator = DALIGenericIterator(data_pipeline, ["data", "steps", "means"])

    data = next(data_iterator)
    sequence = jax.device_put(data['data'], jax.devices('gpu')[0])[0]
    # pred = model(sequence, False)
    # pred_sequential = model(sequence, True)

    timeline = data["steps"][0]
    means = data["means"][0]

    gpus = jax.devices("gpu")
    means = jax.device_put(means, gpus[0])

    delta = sequence[-1]
    rho = cosmos.compute_rho(sequence[-1], means[-1])

    scaling = 10000000
    rho = rho[0]
    rho *= scaling
    print(f"total mass {rho.sum()}")
    print(f"mean {rho.mean()}, max {rho.max()}, min {rho.min()} var {rho.var()}")
    print(rho.shape)
    print(config.grid_size)

    lagrangian_position, euelerian_position, mass = field.fit_field(
        jax.random.PRNGKey(0),
        config.grid_size,
        rho,
        jnp.sum(rho),
        3000,
        learning_rate=0.0001)
    
    mass /= scaling
    
    a = 1 / (1 + config.redshift_start)
    
    #euelerian_position = lagrangian_position + dspls * m_Dplus;
    D_plus = cosmos.growth_factor_approx(a, config.omega_M, config.omega_L)
    
    displacement = (euelerian_position - lagrangian_position) / D_plus

    D_plus_da = cosmos.growth_factor_approx_deriv(config.omega_M, config.omega_L)

    velocity = displacement * D_plus_da

    # normalize to PKDGRAV3 standards
    euelerian_position = euelerian_position - 0.5
    generate_tipsy(
        config.output_tipsy_file,
        euelerian_position,
        velocity,
        mass,
        a)
    
    # Delete Data Pipeline
    del data_pipeline

if __name__ == "__main__":
    main(sys.argv[1:])