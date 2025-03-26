import sys
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from config import Config
from mpl_toolkits.axes_grid1 import make_axes_locatable
from powerbox import get_power
from cosmos import compute_overdensity, to_redshift, normalize, normalize_inv
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
# JAX 
import jax
import jax.numpy as jnp
# NVIDIA Dali
from nvidia.dali.plugin.jax import DALIGenericIterator
# Local
import nn
import data
from config import Config


def plot(
        ouput_file : str,
        config : Config,
        sequence : jax.Array,
        sequence_prediction : jax.Array | None,
        attributes : jax.Array):
    
    plt.rcParams.update({
        'font.size': 28,                   # Global font size
        'axes.labelsize': 20,              # X and Y label font size
        'axes.titlesize': 20,              # Title font size
        'xtick.labelsize': 14,             # X tick label font size
        'ytick.labelsize': 14,             # Y tick label font size
        'legend.fontsize': 18,             # Legend font size
        # 'axes.grid': True,                 # Enable grid
        'grid.alpha': 0.7,                 # Grid line transparency
        'grid.linestyle': '--',            # Grid line style
        'grid.color': 'gray',              # Grid line color
        'text.usetex': False,              # Use TeX for text (set True if TeX is available)
        'figure.figsize': [8, 6],          # Figure size
        # 'axes.prop_cycle': plt.cycler('color', ['#0077BB', '#EE7733', '#33BBEE', '#EE3377'])
    })
    
    frames = sequence.shape[0]
    grid_size = sequence.shape[2]


    t_steps = 10 #config.total_index_steps
    
    # transform to shape for matplotlib
    sequence = jnp.reshape(
        sequence, (frames, grid_size, grid_size, grid_size, 1))

    sequence_prediction = jnp.reshape(
        sequence_prediction, (frames-1, grid_size, grid_size, grid_size, 1))
    
    fig = plt.figure(
        layout='constrained', 
        figsize=(2+2.5*frames, 9 ),
        dpi=300)
    subfigs = fig.subfigures(2, 1, wspace=0.07, hspace=0.05, height_ratios=[1.6, 1])

    spec_sequence = subfigs[0].add_gridspec(2 , frames,  wspace=0.1, hspace=0.0)
    spec_stats = subfigs[1].add_gridspec(1, 2)

    ax_cdf = fig.add_subplot(spec_stats[0], adjustable='box')
    ax_power = fig.add_subplot(spec_stats[1],  adjustable='box')

    cmap = get_cmap('viridis') 
    colors = cmap(jnp.linspace(0, 1, frames))

    file_index_stride = config.file_index_stride.copy()

    if config.flip and isinstance(file_index_stride, list): 
        step = jnp.sum(jnp.array(file_index_stride)) + config.file_index_start
        file_index_stride.reverse()
    elif config.flip:
        step = config.file_index_start + config.file_index_stride * (frames - 1)
    else:
        step = config.file_index_start

    legend_lines = []
    legend_names = []
    for frame in range(frames):
        print(f"step {step}")
        attribs = jax.device_put(attributes[frame], device=jax.devices("gpu")[0])
        normalized = sequence[frame]
        rho = normalize_inv(normalized, attribs, config.normalizing_function)
        delta = compute_overdensity(rho)

        ax_seq = fig.add_subplot(spec_sequence[0, frame])
        ax_seq.set_title(fr'sim $z = {to_redshift(step/t_steps):.2f}$')
        ax_seq.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        im_seq = ax_seq.imshow(normalized[grid_size // 2, : , :], cmap='inferno')
        divider = make_axes_locatable(ax_seq)
        cax = divider.append_axes('right', size='5%', pad=0.03)
        fig.colorbar(im_seq, cax=cax, orientation='vertical')

        ax_cdf.hist(
            normalized.flatten(),
            20, 
            density=True, 
            log=True, 
            histtype="step",
            cumulative=False, 
            linestyle="dashed",
            label=fr'sim $z = {to_redshift(step/t_steps):.2f}$',
            color=colors[frame])
        
        p,k = get_power(delta[:, :, :, 0], config.box_size)
        ax_power.plot(
            k, 
            p, 
            label=fr'sim $z = {to_redshift(step/t_steps):.2f}$',
            linestyle="dashed",
            color=colors[frame])
        
        legend_lines.append(
            Line2D([0], [0], color=colors[frame], lw=2, linestyle="dashed")
        )
        legend_names.append(fr'sim $z = {to_redshift(step/t_steps):.2f}$')
        
        if frame < frames-1:
            if isinstance(file_index_stride, list): 
                step += file_index_stride[frame] * (-1 if config.flip else 1)
            else:
                step += file_index_stride * (-1 if config.flip else 1)

        if frame < frames-1:
            rho_pred_normalized = sequence_prediction[frame]
            attribs = jax.device_put(attributes[frame+1], device=jax.devices("gpu")[0])
            rho_pred = normalize_inv(rho_pred_normalized, attribs, config.normalizing_function)
            delta_pred = compute_overdensity(rho_pred)

            normalized = sequence[frame+1]
            rho = normalize_inv(normalized, attribs, config.normalizing_function)
            delta = compute_overdensity(rho_pred)

            ax_seq = fig.add_subplot(spec_sequence[1, frame+1])
            ax_seq.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            ax_seq.set_title(fr'pred $z = {to_redshift(step/t_steps):.2f}$')

            im_seq = ax_seq.imshow(rho_pred_normalized[grid_size // 2, : , :], cmap='inferno')
            divider = make_axes_locatable(ax_seq)
            cax = divider.append_axes('right', size='5%', pad=0.03)
            fig.colorbar(im_seq, cax=cax, orientation='vertical')

            ax_cdf.hist(
                rho_pred_normalized.flatten(),
                100, 
                density=True, 
                log=True, 
                histtype="step",
                cumulative=False, 
                label=fr'pred $z = {to_redshift(step/t_steps):.2f}$',
                color=colors[frame+1])
            
            p,k = get_power(delta_pred[:, :, :, 0], config.box_size)
            ax_power.plot(
                k, p, 
                label=fr'pred $z = {to_redshift(step/t_steps):.2f}$',
                color=colors[frame+1])
            
            legend_lines.append(
                Line2D([0], [0], color=colors[frame+1], lw=4)
            )
            legend_names.append(fr'pred $z = {to_redshift(step/t_steps):.2f}$')

    ax_legend = fig.add_subplot(spec_sequence[1, 0])
    ax_legend.legend(
        legend_lines, legend_names,  loc='center'
    )
    ax_legend.spines['top'].set_visible(False)
    ax_legend.spines['right'].set_visible(False)
    ax_legend.spines['bottom'].set_visible(False)
    ax_legend.spines['left'].set_visible(False)
    ax_legend.set_xticks([])
    ax_legend.set_yticks([])

    ax_power.set_yscale('log')
    ax_power.set_xscale('log')
    #ax_power.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax_power.set_title(r'Power Spectrum of $\delta$')
    ax_power.set_xlabel(r'$k$ [$h \ \mathrm{Mpc}^{-1}$]')
    ax_power.set_ylabel(r'$P(k)$ [$h^{-3} \ \mathrm{Mpc}^3$]')

    ax_cdf.set_title(r'pdf $\rho_{norm}$')
    #ax_cdf.legend(loc='lower center', bbox_to_anchor=(0.5, 1.745), fancybox=True, shadow=True,)

    plt.savefig(ouput_file)

def main(argv) -> None:
   
    model_name = argv[0]

    model, config, training_stats = nn.load_sequential_model(
        model_name)
    
    # Data Pipeline
    dataset = data.DirectorySequence(
        grid_size = config.input_grid_size,
        grid_directory = config.grid_dir,
        start = config.file_index_start,
        steps = config.file_index_steps,
        stride = config.file_index_stride,
        normalizing_function = config.normalizing_function,
        flip = config.flip,        
        type = "test")  

    data_pipeline = data.directory_sequence_pipe(dataset, config.grid_size)
    data_iterator = DALIGenericIterator(data_pipeline, ["data", "attributes"])

    sample = next(data_iterator)
    sequence = jax.device_put(sample['data'], jax.devices('gpu')[0])[0]
    attributes = jax.device_put(sample['attributes'], jax.devices('gpu')[0])[0]
    pred = model(sequence, attributes, False, config.include_potential)
    pred_seq = model(sequence, attributes, True, config.include_potential)

    attributes = sample["attributes"][0]

    plot(
        "img/prediction_stepwise.jpg", 
        sequence = sequence, 
        config = config,
        sequence_prediction = pred[:, 0:1],
        attributes = attributes)

    plot(
        "img/prediction_sequential.jpg", 
        sequence = sequence, 
        config = config,
        sequence_prediction = pred_seq[:, 0:1],
        attributes = attributes)

    # Delete Data Pipeline
    del data_pipeline

if __name__ == "__main__":
    main(sys.argv[1:])