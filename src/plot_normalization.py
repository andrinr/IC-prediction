import sys
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from powerbox import get_power
from nvidia.dali.plugin.jax import DALIGenericIterator

import data
from config import load_config
from cosmos import compute_overdensity, normalize_inv, to_redshift

def main(argv) -> None:

    config = load_config(argv[0])   

    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_disable_jit", False)

    dataset_params = {
        "grid_size" : config.input_grid_size,
        "grid_directory" : config.grid_dir,
        "start" : config.file_index_start,
        "steps" : config.file_index_steps,
        "stride" : config.file_index_stride,
        "normalizing_function" : config.normalizing_function,
        "flip" : config.flip,
        "type" : "test"}

    dataset = data.DirectorySequence(**dataset_params)
    data_pipeline = data.directory_sequence_pipe(dataset, config.grid_size)
    data_iterator = DALIGenericIterator(data_pipeline, ["data", "attributes"])

    sample = next(data_iterator)
    sequence = jax.device_put(sample['data'], jax.devices('gpu')[0])[0]

    attributes = sample["attributes"][0]

    # Plotting
    plt.rcParams.update({
        'font.size': 28,                   # Global font size
        'axes.labelsize': 24,              # X and Y label font size
        'axes.titlesize': 24,              # Title font size
        'xtick.labelsize': 14,             # X tick label font size
        'ytick.labelsize': 14,             # Y tick label font size
        'legend.fontsize': 22,             # Legend font size
        # 'axes.grid': True,                 # Enable grid
        'grid.alpha': 0.7,                 # Grid line transparency
        'grid.linestyle': '--',            # Grid line style
        'grid.color': 'gray',              # Grid line color
        'text.usetex': False,              # Use TeX for text (set True if TeX is available)
        'figure.figsize': [8, 6],          # Figure size
        # 'axes.prop_cycle': plt.cycler('color', ['#0077BB', '#EE7733', '#33BBEE', '#EE3377'])
    })

    n_steps = sequence.shape[0]
    grid_size = sequence.shape[2]

    fig = plt.figure(
        layout='constrained', 
        figsize=(4 + 3.5 * n_steps, 10),
        dpi=300)
    
    subfigs = fig.subfigures(
        2, 
        1, 
        wspace=0.01, 
        hspace=0.01)
    
    spec_sequence = subfigs[0].add_gridspec(2, n_steps, wspace=0.1, hspace=0.1)
    spec_stats = subfigs[1].add_gridspec(1, 2)

    ax_power = fig.add_subplot(spec_stats[0], adjustable='box')
    ax_cdf = fig.add_subplot(spec_stats[1], adjustable='box')

    sequence = jnp.reshape(sequence, (n_steps, grid_size, grid_size, grid_size, 1))

    step = config.file_index_start

    if config.flip and isinstance(config.file_index_stride, list): 
        step = jnp.sum(jnp.array(config.file_index_stride)) + config.file_index_start
        config.file_index_stride.reverse()
    elif config.flip:
        step = config.file_index_start + config.file_index_stride * (n_steps - 1)
    else:
        step = config.file_index_start

    for i in range(n_steps):

        grid_size = sequence[i].shape[2]
        
        attribs = jax.device_put(attributes[i], device=jax.devices("gpu")[0])
        rho = normalize_inv(sequence[i], attribs, config.normalizing_function)
        delta = compute_overdensity(rho)

        ax_seq = fig.add_subplot(spec_sequence[0:2, i])
        ax_seq.set_title(r'$\rho_{norm}$' + fr' $z={to_redshift(step/100):.2f}$')
        ax_seq.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        im_seq = ax_seq.imshow(sequence[i][grid_size // 2, :, :], cmap='inferno')
        divider = make_axes_locatable(ax_seq)
        cax = divider.append_axes('bottom', size='5%', pad=0.03)
        fig.colorbar(im_seq, cax=cax, orientation='horizontal')

        p_pred, k_pred = get_power(delta[:, :, :, 0], config.box_size)
        ax_power.plot(
            k_pred, p_pred,
            label=fr'$z={to_redshift(step/100):.2f}$',
            linewidth=2)
        
        ax_cdf.hist(
            sequence[i].flatten(),
            100,
            density=True,
            log=True,
            histtype="step",
            cumulative=False,
            label=fr'$z = {to_redshift(step/100):.2f}$',
            linewidth=2)

        if i < n_steps-1:
            if isinstance(config.file_index_stride, list): 
                step += config.file_index_stride[i] * (-1 if config.flip else 1)
            else:
                step += config.file_index_stride * (-1 if config.flip else 1)

    ax_power.set_yscale('log')
    ax_power.set_xscale('log')
    ax_power.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax_power.set_title(r'Power Spectrum of $\delta$')
    ax_power.set_xlabel(r'$k$ [$h \ \mathrm{Mpc}^{-1}$]')
    ax_power.set_ylabel(r'$P(k)$ [$h^{-3} \ \mathrm{Mpc}^3$]')
    ax_cdf.set_title(r'pdf $\rho_{norm}$')
    ax_cdf.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(f"img/{config.normalizing_function}.png", bbox_inches="tight", dpi=300)

    del data_pipeline

if __name__ == "__main__":
    main(sys.argv[1:])