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
import field

def main(argv) -> None:
   
    model_name = argv[0]

    model, config, training_stats = nn.load_sequential_model(
        model_name)
    
    # Data Pipeline
    dataset = DirectorySequence(
        grid_size = config.input_grid_size,
        grid_directory = config.grid_dir,
        start = config.file_index_start,
        steps = config.file_index_steps,
        stride = config.file_index_stride,
        normalizing_function = "log_growth",
        flip = config.flip,        
        type = "test")  

    data_pipeline = directory_sequence_pipe(dataset, config.grid_size)
    data_iterator = DALIGenericIterator(data_pipeline, ["data", "attributes"])

    sample = next(data_iterator)
    sequence = jax.device_put(sample['data'], jax.devices('gpu')[0])[0]
    attributes = jax.device_put(sample['attributes'], jax.devices('gpu')[0])[0]
    # pred = model(sequence, False)
    # pred_sequential = model(sequence, True)

    IC = sequence[-1]
    IC_attr = attributes[-1]

    rho = cosmos.normalize_inv(IC, IC_attr, "log_growth" )

    print(IC.shape)
    print(IC_attr.shape)
    print(rho.shape)

    scaling = 0.00001
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
        10000,
        learning_rate=0.0001)
    
    mass /= scaling
    
    a = 1.0
    
    #euelerian_position = lagrangian_position + dspls * m_Dplus;
    D_plus = cosmos.growth_factor_approx(a, config.omega_M, config.omega_L)
    
    displacement = (euelerian_position - lagrangian_position) / D_plus

    D_plus_da = cosmos.growth_factor_approx_deriv(config.omega_M, config.omega_L)

    velocity = displacement * D_plus_da

    # normalize to PKDGRAV3 standards
    euelerian_position = euelerian_position - 0.5
    generate_tipsy(
        "test.tipsy",
        euelerian_position,
        velocity,
        mass,
        a)
    
    # Delete Data Pipeline
    del data_pipeline

if __name__ == "__main__":
    main(sys.argv[1:])