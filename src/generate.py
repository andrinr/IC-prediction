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
import matplotlib.pyplot as plt
from field import cic_ma
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
        normalizing_function = config.normalizing_function,
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
        5000,
        learning_rate=0.0001)
    
    rho /= scaling
    mass /= scaling

    
    L = rho.shape[0]
    rho_pred = cic_ma(
        euelerian_position,
        mass,
        rho.shape[0])
    
    diff = rho - rho_pred
    diff /= rho.std()
    diff = jnp.reshape(diff, (L, L, L, 1))
    fig, axs = plt.subplots()
    im = axs.imshow(diff[L//2])
    divider = make_axes_locatable(axs)
    cax = divider.append_axes('bottom', size='5%', pad=0.03)
    fig.colorbar(im, cax=cax, orientation='horizontal')

    print(rho.mean())
    print(rho_pred.mean())

    plt.savefig("img/cmp.jpg")
    
    a = 1.0
    
    #euelerian_position = lagrangian_position + dspls * m_Dplus;
    D_plus = cosmos.growth_factor_approx(a, config.omega_M, config.omega_L)
    
    # print(D_plus)
    displacement = (euelerian_position - lagrangian_position) / D_plus

    D_plus_da = cosmos.growth_factor_approx_deriv(config.omega_M, config.omega_L)

    # print(D_plus_da)
    # print(displacement.mean())
    velocity = displacement * D_plus_da

    # print(velocity.mean())

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