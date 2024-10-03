import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from config import Config
from mpl_toolkits.axes_grid1 import make_axes_locatable
from powerbox import get_power
from cosmos import compute_overdensity, to_redshift, normalize, normalize_inv
from matplotlib.cm import get_cmap

def sequence(
        ouput_file : str,
        config : Config,
        sequence : jax.Array,
        sequence_prediction : jax.Array | None,
        attributes : jax.Array):
    
    frames = sequence.shape[0]
    grid_size = sequence.shape[2]

    pred = sequence_prediction is not None
    long = frames > 3

    # transform to shape for matplotlib
    sequence = jnp.reshape(
        sequence, (frames, grid_size, grid_size, grid_size, 1))
    if pred:
        sequence_prediction = jnp.reshape(
            sequence_prediction, (frames-1, grid_size, grid_size, grid_size, 1))
    
    fig = plt.figure(layout='constrained', figsize=(4+3*frames, 10 if pred else 7))
    subfigs = fig.subfigures(2, 1, wspace=0.07, hspace=0.1, height_ratios=[2, 1] if pred else [1, 1] )

    spec_sequence = subfigs[0].add_gridspec(2 if pred else 1, frames,  wspace=0.3, hspace=0.1)
    spec_stats = subfigs[1].add_gridspec(1, 4 if pred and long else 2)

    ax_cdf = fig.add_subplot(spec_stats[0], adjustable='box', aspect=0.1)
    ax_power = fig.add_subplot(spec_stats[2 if pred and long else 1],  adjustable='box', aspect=0.1)
    if pred and long:
        ax_cdf_pred = fig.add_subplot(spec_stats[1], adjustable='box', aspect=0.1)
        ax_power_pred = fig.add_subplot(spec_stats[3], adjustable='box', aspect=0.1)

    cmap = get_cmap('viridis') if long else get_cmap('Accent')
    colors = cmap(jnp.linspace(0, 1, frames if long else 6))

    file_index_stride = config.file_index_stride

    if config.flip and isinstance(file_index_stride, list): 
        step = jnp.sum(jnp.array(file_index_stride))
        file_index_stride.reverse()
    elif config.flip:
        step = config.file_index_start + config.file_index_stride * frames - 1
    else:
        step = config.file_index_start

    for frame in range(frames):
        attribs = jax.device_put(attributes[frame], device=jax.devices("gpu")[0])
        normalized = sequence[frame]
        rho = normalize_inv(normalized, attribs, config.normalizing_function)
        delta = compute_overdensity(rho)

        ax_seq = fig.add_subplot(spec_sequence[0, frame])
        ax_seq.set_title(fr'sim $z = {to_redshift(step/100):.2f}$')
        ax_seq.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        im_seq = ax_seq.imshow(normalized[grid_size // 2, : , :], cmap='inferno')
        divider = make_axes_locatable(ax_seq)
        cax = divider.append_axes('bottom', size='5%', pad=0.03)
        fig.colorbar(im_seq, cax=cax, orientation='horizontal')

        ax_cdf.hist(
            normalized.flatten(),
            20, 
            density=True, 
            log=True, 
            histtype="step",
            cumulative=False, 
            label=fr'sim $z = {to_redshift(step/100):.2f}$',
            color=colors[frame])
        
        p,k = get_power(delta[:, :, :, 0], config.box_size)
        ax_power.plot(
            k, 
            p, 
            label=fr'sim $z = {to_redshift(step/100):.2f}$',
            color=colors[frame])
        
        if frame < frames-1:
            if isinstance(file_index_stride, list): 
                step += file_index_stride[frame] * (-1 if config.flip else 1)
            else:
                step += file_index_stride * (-1 if config.flip else 1)

        if pred and frame < frames-1:
            rho_pred_normalized = sequence_prediction[frame]
            attribs = jax.device_put(attributes[frame+1], device=jax.devices("gpu")[0])
            rho_pred = normalize_inv(rho_pred_normalized, attribs, config.normalizing_function)
            delta_pred = compute_overdensity(rho_pred)

            ax_seq = fig.add_subplot(spec_sequence[1, frame+1])
            ax_seq.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            ax_seq.set_title(fr'pred $z = {to_redshift(step/100):.2f}$')

            im_seq = ax_seq.imshow(rho_pred_normalized[grid_size // 2, : , :], cmap='inferno')
            divider = make_axes_locatable(ax_seq)
            cax = divider.append_axes('bottom', size='5%', pad=0.03)
            fig.colorbar(im_seq, cax=cax, orientation='horizontal')

            axis = ax_cdf_pred if long else ax_cdf
            axis.hist(
                rho_pred_normalized.flatten(),
                100, 
                density=True, 
                log=True, 
                histtype="step",
                cumulative=False, 
                label=fr'pred $z = {to_redshift(step/100):.2f}$',
                color=colors[frame+1 if long else 3 + frame])
            
            axis = ax_power_pred if long else ax_power
            p,k = get_power(delta_pred[:, :, :, 0], config.box_size)
            axis.plot(
                k, p, 
                label=fr'pred $z = {to_redshift(step/100):.2f}$',
                color=colors[frame+1 if long else 3 + frame])

    ax_power.set_yscale('log')
    ax_power.set_xscale('log')
    ax_power.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax_power.set_title(r'Power Spectrum of $\delta$')
    ax_power.set_xlabel(r'$k$ [$h \ \mathrm{Mpc}^{-1}$]')
    ax_power.set_ylabel(r'$P(k)$ [$h^{-3} \ \mathrm{Mpc}^3$]')

    if pred and long:
        ax_power_pred.set_yscale('log')
        ax_power_pred.set_xscale('log')
        ax_power_pred.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax_power_pred.set_title(r'Power Spectrum pred $\delta$')
        ax_power_pred.set_xlabel(r'$k$ [$h \ \mathrm{Mpc}^{-1}$]')
        ax_power_pred.set_ylabel(r'$P(k)$ [$h^{-3} \ \mathrm{Mpc}^3$]')

        ax_cdf_pred.set_title(r'pdf $\rho_{norm}$')
        ax_cdf_pred.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax_cdf.set_title(r'pdf $\rho_{norm}$')
    ax_cdf.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.97], pad=3.0, w_pad=2.0, h_pad=2.0)

    plt.savefig(ouput_file)