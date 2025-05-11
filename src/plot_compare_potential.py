import sys
import os
import jax
import jax.numpy as jnp
from nvidia.dali.plugin.jax import DALIGenericIterator
import nn
import data
from config import Config
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from powerbox import get_power
from cosmos import compute_overdensity, to_redshift, normalize_inv

def compare(
        output_file: str,
        config: Config,
        sequences: list[jax.Array],
        predictions: list[jax.Array],
        labels : list[str],
        attributes: list[jax.Array],
        norm_functions : list[str]):

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

    
    num_predictions = len(predictions)  

    # Create figure
    fig = plt.figure(
        layout='constrained', 
        figsize=(4 + 4 * num_predictions, 5),  
        dpi=300)

    subfigs = fig.subfigures(
        1, 
        2, 
        wspace=0.05, 
        hspace=0.01,
        width_ratios=[2, 1.5],
        height_ratios=[1])

    spec_sequence = subfigs[0].add_gridspec(
        2, 
        num_predictions + 1, 
        wspace=0.2, 
        hspace=0.00)
    spec_stats = subfigs[1].add_gridspec(3, 1)
    ax_power = fig.add_subplot(spec_stats[0:2], adjustable='box')

    file_index_stride = config.file_index_stride
    step = config.file_index_start

    frames = sequences[0].shape[0]
    grid_size = sequences[0].shape[2]
        
    sequence_curr = jnp.reshape(sequences[0], (frames, grid_size, grid_size, grid_size, 1))
    attributes_curr = attributes[0]
    attribs = jax.device_put(attributes_curr[1], device=jax.devices("gpu")[0])
    normalized = sequence_curr[-1]
    rho = normalize_inv(normalized, attribs, norm_functions[0])
    delta = compute_overdensity(rho)

    step = config.file_index_start

    ax_seq = fig.add_subplot(spec_sequence[1, 0])
    ax_seq.set_title(r'$\rho_{norm}$' + fr' $z={to_redshift(step/100):.2f}$')
    ax_seq.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    im_seq = ax_seq.imshow(normalized[grid_size // 2, :, :], cmap='inferno')
    divider = make_axes_locatable(ax_seq)
    cax = divider.append_axes('right', size='5%', pad=0.03)
    fig.colorbar(im_seq, cax=cax, orientation='vertical')

    p, k = get_power(delta[:, :, :, 0], config.box_size)
    ax_power.plot(
        k,
        p,
        label=fr'sim $z = {to_redshift(step/100):.2f}$')

    if isinstance(file_index_stride, list): 
        step = jnp.sum(jnp.array(file_index_stride)) + config.file_index_start
        file_index_stride.reverse()
    else:
        step = config.file_index_start + config.file_index_stride * (frames - 1)

    normalized = sequence_curr[0]
    ax_seq = fig.add_subplot(spec_sequence[0, 0])
    ax_seq.set_title(r'$\rho_{norm}$' + fr' $z={to_redshift(step/100):.2f}$')
    ax_seq.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    im_seq = ax_seq.imshow(normalized[grid_size // 2, :, :], cmap='inferno')
    divider = make_axes_locatable(ax_seq)
    cax = divider.append_axes('right', size='5%', pad=0.03)
    fig.colorbar(im_seq, cax=cax, orientation='vertical')
    
    for idx in range(num_predictions):

        frames = sequences[idx].shape[0]
        grid_size = sequences[idx].shape[2]
        
        sequence_curr = jnp.reshape(sequences[idx], (frames, grid_size, grid_size, grid_size, 1))
        pred_curr = jnp.reshape(predictions[idx], (frames - 1, grid_size, grid_size, grid_size, 1))
        attributes_curr = attributes[idx]
        
        attribs = jax.device_put(attributes_curr[1], device=jax.devices("gpu")[0])
        normalized = sequence_curr[1]
        rho = normalize_inv(normalized, attribs, norm_functions[idx])
        delta = compute_overdensity(rho)

        rho_pred_normalized = pred_curr[0]
        rho_pred = normalize_inv(rho_pred_normalized, attribs,  norm_functions[idx])
        delta_pred = compute_overdensity(rho_pred)

        normalized = sequence_curr[1]
        ax_seq = fig.add_subplot(spec_sequence[0, idx+1])
        ax_seq.set_title(r'$\hat{\rho}_{norm} - \rho_{norm}$')
        ax_seq.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        im_seq = ax_seq.imshow(
            rho_pred_normalized[grid_size // 2, :, :] - normalized[grid_size // 2, :, :], 
            cmap='RdYlBu',
            vmin=-0.15,
            vmax=0.15)

        divider = make_axes_locatable(ax_seq)
        cax = divider.append_axes('right', size='5%', pad=0.03)
        fig.colorbar(
            im_seq, 
            cax=cax, 
            orientation='vertical')
    
        ax_seq_pred = fig.add_subplot(spec_sequence[1, idx+1])
        ax_seq_pred.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax_seq_pred.set_title(r'$\hat{\rho}_{norm}$' + f' {labels[idx]}')
        im_seq_pred = ax_seq_pred.imshow(rho_pred_normalized[grid_size // 2, :, :], cmap='inferno')
        
        divider_pred = make_axes_locatable(ax_seq_pred)
        cax_pred = divider_pred.append_axes('right', size='5%', pad=0.03)
        fig.colorbar(im_seq_pred, cax=cax_pred, orientation='vertical')
        
        p_pred, k_pred = get_power(delta_pred[:, :, :, 0], config.box_size)
        ax_power.plot(
            k_pred, p_pred,
            label=fr'pred {labels[idx]}')
        
    ax_power.set_yscale('log')
    ax_power.set_xscale('log')
    ax_power.legend(loc='center left', bbox_to_anchor=(0, -0.9))
    ax_power.set_title(r'Power Spectrum of $\delta$')
    ax_power.set_xlabel(r'$k$ [$h \ \mathrm{Mpc}^{-1}$]')
    ax_power.set_ylabel(r'$P(k)$ [$h^{-3} \ \mathrm{Mpc}^3$]')

    plt.savefig(output_file, dpi=300)

def main(argv) -> None:

    folder = argv[0]

    files = os.listdir(folder)
    files.sort()

    inputs = []
    predictions = []
    attributes_ls = []
    labels = []
    norm_functions = []

    # load config from first model, assuming data configs are all identical
    for i, file in enumerate(files):

        filename = os.path.join(folder, file)
        model, config, _ = nn.load_sequential_model(filename)

        labels.append("+potential" if i == 1 else "")

        norm_functions.append(config.normalizing_function)

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
        sequence = jax.device_put(sample['data'], jax.devices('gpu')[0])[1]
        attributes = jax.device_put(sample['attributes'], jax.devices('gpu')[0])[1]
        
        pred_sequential = model(sequence, attributes, False, config.include_potential)

        predictions.append(pred_sequential)
        inputs.append(sequence)
        attributes_ls.append(attributes)

    compare(
        "img/compare.png", 
        sequences = inputs, 
        config = config,
        predictions = predictions,
        attributes = attributes_ls,
        labels = labels,
        norm_functions = norm_functions)

    # Delete Data Pipeline
    del data_pipeline

if __name__ == "__main__":
    main(sys.argv[1:])