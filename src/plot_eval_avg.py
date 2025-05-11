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
        attributes : jax.Array,
        training_stats : dict):
    
    plt.rcParams.update({
        'font.size': 28,                   # Global font size
        'axes.labelsize': 20,              # X and Y label font size
        'axes.titlesize': 20,              # Title font size
        'xtick.labelsize': 14,             # X tick label font size
        'ytick.labelsize': 14,             # Y tick label font size
        'legend.fontsize': 18,             # Legend font size
        'grid.alpha': 0.7,                 # Grid line transparency
        'grid.linestyle': '--',            # Grid line style
        'grid.color': 'gray',              # Grid line color
        'text.usetex': False,              # Use TeX for text (set True if TeX is available)
    })
    
    frames = sequence.shape[1]
    grid_size = sequence.shape[3]
    shots = sequence.shape[0]

    print(training_stats)


    t_steps = 10 #config.total_index_steps
    
    # transform to shape for matplotlib
    sequence = jnp.reshape(
        sequence, (shots, frames, grid_size, grid_size, grid_size, 1))

    sequence_prediction = jnp.reshape(
        sequence_prediction, (shots, frames-1, grid_size, grid_size, grid_size, 1))
    
    fig = plt.figure(
        layout='constrained', 
        figsize=(3+2*shots // 2, 9 ),
        dpi=300)
    subfigs = fig.subfigures(2, 1, wspace=0.00, hspace=0.05, height_ratios=[1.6, 1])

    spec_sequence = subfigs[0].add_gridspec(1 , 3,  wspace=0.0, hspace=0.06)
    spec_stats = subfigs[1].add_gridspec(1, 2)

    ax_corr = fig.add_subplot(spec_stats[0], adjustable='box')
    ax_loss = fig.add_subplot(spec_stats[1],  adjustable='box')

    cmap = get_cmap('viridis') 
    colors = cmap(jnp.linspace(0, 1, frames))

    if config.flip and isinstance(config.file_index_stride, list): 
        file_index_stride = config.file_index_stride.copy()
        step = jnp.sum(jnp.array(file_index_stride)) + config.file_index_start
        file_index_stride.reverse()
    elif config.flip:
        step = config.file_index_start + config.file_index_stride * (frames - 1)
    else:
        step = config.file_index_start

    ps = []
    ps_true = []
    corr = []
    preds = []

    for shot in range(shots):
        rho_pred_normalized = sequence_prediction[shot][-1]
        preds.append(rho_pred_normalized)
        attribs = jax.device_put(attributes[shot][-1], device=jax.devices("gpu")[0])
        rho_pred = normalize_inv(rho_pred_normalized, attribs, config.normalizing_function)
        delta_pred = compute_overdensity(rho_pred)

        normalized = sequence[shot][-1]
        rho = normalize_inv(normalized, attribs, config.normalizing_function)
        delta = compute_overdensity(rho)

        normalized_from = sequence[shot][0]

        p,k = get_power(delta_pred[:, :, :, 0], config.box_size)

        if shot == 3:
            ax_seq = fig.add_subplot(spec_sequence[0, 0])
            ax_seq.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            ax_seq.set_title(fr'sim $z = 0$')
            im_seq = ax_seq.imshow(normalized_from[grid_size // 2, : , :], cmap='inferno')
            divider = make_axes_locatable(ax_seq)
            cax = divider.append_axes('bottom', size='2%', pad=0.03)
            fig.colorbar(im_seq, cax=cax, orientation='horizontal')
        
            ax_seq = fig.add_subplot(spec_sequence[0, 1])
            ax_seq.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            ax_seq.set_title(fr'pred $z = 49$')
            im_seq = ax_seq.imshow(rho_pred_normalized[grid_size // 2, : , :], cmap='inferno')
            divider = make_axes_locatable(ax_seq)
            cax = divider.append_axes('bottom', size='2%', pad=0.03)
            fig.colorbar(im_seq, cax=cax, orientation='horizontal')

            ax_seq = fig.add_subplot(spec_sequence[0, 2])
            ax_seq.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            ax_seq.set_title(fr'sim $z = 49$')
            im_seq = ax_seq.imshow(normalized[grid_size // 2, : , :], cmap='inferno')
            divider = make_axes_locatable(ax_seq)
            cax = divider.append_axes('bottom', size='2%', pad=0.03)
            fig.colorbar(im_seq, cax=cax, orientation='horizontal')

        p_true, k = get_power(
            delta[:, :, :, 0],
            config.box_size)

        p_pred, k_pred = get_power(
            delta[:, :, :, 0],
            config.box_size,
            delta_pred[:, :, :, 0])
        
        ps.append(p)
        corr.append(p_pred)
        ps_true.append(p_true)
      
    ps = jnp.array(ps)
    corr = jnp.array(corr)

    mean_ps = jnp.mean(ps, axis=0)

    mean_corr = jnp.mean(corr, axis=0)

    std_ps = jnp.std(ps, axis=0)
    min_corr = jnp.min(corr, axis=0)
    max_corr = jnp.max(corr, axis=0)


    ax_corr.plot(
        k_pred,
        mean_ps,
        label=r'$\langle |\delta(k)_{pred}|^2 \rangle$',
        linestyle="solid",
        color='red')
    
    ax_corr.fill_between(
        k_pred,
        mean_ps - std_ps,
        mean_ps + std_ps,
        alpha=0.2,
        color='red')

    ax_corr.plot(
        k_pred,
        mean_corr,
        label=r'$\langle |\delta(k)| \cdot |\delta(k)_{pred}| \rangle$',
        linestyle="solid",
        color='black')
    
    ax_corr.plot(
        k,
        p_true,
        label=r'$\langle |\delta(k)|^2 \rangle$',
        linestyle="solid",
        color='blue')

    ax_corr.fill_between(
        k_pred,
        min_corr,
        max_corr,
        alpha=0.2,
        color='black')
    ax_corr.set_yscale('log')
    ax_corr.set_xscale('log')
    # ax_corr.set_title(r'Cross Correlatiopn of $\delta$')
    ax_corr.set_xlabel(r'$k$ [$h \ \mathrm{Mpc}^{-1}$]')
    ax_corr.set_ylabel(r'$P(k)$ [$h^{-3} \ \mathrm{Mpc}^3$]')
    ax_corr.legend(loc='lower left')

    for i in range(shots):
        corr_ = corr[i] / jnp.sqrt(ps[i] * ps_true[i])

        ax_loss.plot(
            k_pred,
            corr_ * 100,
            label=r'$\langle |\delta(k)| \cdot |\delta(k)_{pred}| \rangle$ / ($\langle |\delta(k)|^2 \rangle$ * $\langle |\delta(k)_{pred}|^2 \rangle)$',
            linestyle="solid",
            color="black",
            alpha=0.2)
        
    ax_loss.set_xscale('log')
    ax_loss.set_xlabel(r'$k$ [$h \ \mathrm{Mpc}^{-1}$]')
    ax_loss.set_ylabel(r'$percent$ [%]')
    # ax_loss.legend(loc='upper right')
    
    plt.savefig(ouput_file)

def main(argv) -> None:
   
    model_name = argv[0]

    model, config, training_stats = nn.load_sequential_model(
        model_name)

    n_shots = 6
    
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

    predictions = []
    attributes = []
    sequences = []
    for i in range(n_shots):
        sample = next(data_iterator)
        batch_size = sample['data'].shape[0]
        for j in range(batch_size):
            sequence = jax.device_put(sample['data'], jax.devices('gpu')[0])[j]
            attribute = jax.device_put(sample['attributes'], jax.devices('gpu')[0])[j]

            pred_seq = model(sequence, attribute, True, config.include_potential)

            predictions.append(pred_seq)
            attributes.append(attribute)
            sequences.append(sequence)

    predictions = jnp.array(predictions)
    attributes = jnp.array(attributes)
    sequences = jnp.array(sequences)

    plot(
        "img/prediction_sequential.jpg", 
        sequence = sequences, 
        config = config,
        sequence_prediction = predictions,
        attributes = attributes,
        training_stats = training_stats)

    # Delete Data Pipeline
    del data_pipeline

if __name__ == "__main__":
    main(sys.argv[1:])