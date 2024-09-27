import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from config import Config
from cosmos import PowerSpectrum
from mpl_toolkits.axes_grid1 import make_axes_locatable
from powerbox import PowerBox, get_power
from data import normalize_inv
from cosmos import compute_overdensity
from matplotlib.cm import get_cmap

def sequence(
        ouput_file : str,
        config : Config,
        sequence : jax.Array,
        sequence_prediction : jax.Array | None,
        timeline : jax.Array,
        attributes : jax.Array):
    
    frames = sequence.shape[0]
    grid_size = sequence.shape[2]

    pred = sequence_prediction is not None

    # transform to shape for matplotlib
    sequence = jnp.reshape(
        sequence, (frames, grid_size, grid_size, grid_size, 1))
    if pred:
        sequence_prediction = jnp.reshape(
            sequence_prediction, (frames-1, grid_size, grid_size, grid_size, 1))
    
    fig = plt.figure(layout='constrained', figsize=(4+3*frames, 10 if pred else 7))
    subfigs = fig.subfigures(2, 1, wspace=0.07, height_ratios=[2, 1])

    spec_sequence = subfigs[0].add_gridspec(2 if pred else 1, frames)
    spec_stats = subfigs[1].add_gridspec(1, 4 if pred else 2)

    ax_cdf = fig.add_subplot(spec_stats[0])
    ax_power = fig.add_subplot(spec_stats[2 if pred else 1])
    if pred:
        ax_cdf_pred = fig.add_subplot(spec_stats[1])
        ax_power_pred = fig.add_subplot(spec_stats[3])

    cmap = get_cmap('viridis') 
    colors = cmap(jnp.linspace(0, 1, frames))

    for frame in range(frames):

        min = attributes[frame, 0]
        max = attributes[frame, 1]

        min = jax.device_put(min, device=jax.devices("gpu")[0])
        max = jax.device_put(max, device=jax.devices("gpu")[0])

        rho_normalized = sequence[frame]
        rho = normalize_inv(rho_normalized, min, max)
        delta, mean = compute_overdensity(rho)

        ax_seq = fig.add_subplot(spec_sequence[0, frame])
        ax_seq.set_title(f"sim t: {timeline[frame]}")

        im_seq = ax_seq.imshow(rho_normalized[grid_size // 2, : , :], cmap='inferno')
        divider = make_axes_locatable(ax_seq)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im_seq, cax=cax, orientation='vertical')

        if pred and frame < frames-1:
            rho_pred_normalized = sequence_prediction[frame]
            rho_pred = normalize_inv(rho_pred_normalized, min, max)
            delta_pred, mean_pred = compute_overdensity(rho_pred)

            ax_seq = fig.add_subplot(spec_sequence[1, frame+1])
            ax_seq.set_title(f"pred t: {timeline[frame+1]}")

            im_seq = ax_seq.imshow(rho_pred_normalized[grid_size // 2, : , :], cmap='inferno')
            divider = make_axes_locatable(ax_seq)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im_seq, cax=cax, orientation='vertical')

            ax_cdf_pred.hist(
                rho_pred_normalized.flatten(),
                100, 
                density=True, 
                log=True, 
                histtype="step",
                cumulative=False, 
                label=f"pred t: {timeline[frame+1]}", 
                color=colors[frame+1])
            
            p,k = get_power(delta_pred[:, :, :, 0], config.box_size)
            ax_power_pred.plot(
                k, p, 
                label=f"pred t: {timeline[frame+1]}",  
                color=colors[frame+1])

        ax_cdf.hist(
            rho_normalized.flatten(), 
            100, 
            density=True, 
            log=True, 
            histtype="step",
            cumulative=False, 
            label=f"sim t: {timeline[frame]}", 
            color=colors[frame])
        
        p,k = get_power(delta[:, :, :, 0], config.box_size)
        ax_power.plot(
            k, 
            p, 
            label=f"sim t: {timeline[frame]}", 
            color=colors[frame])

    ax_power.set_yscale('log')
    ax_power.set_xscale('log')
    ax_power.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax_power.set_title(r'Power Spectrum simulated $\delta$')
    ax_power.set_xlabel(r'$k$ [$h \ \mathrm{Mpc}^{-1}$]')
    ax_power.set_ylabel(r'$P(k)$ [$h^{-3} \ \mathrm{Mpc}^3$]')

    if pred:
        ax_power_pred.set_yscale('log')
        ax_power_pred.set_xscale('log')
        ax_power_pred.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax_power_pred.set_title(r'Power Spectrum predicted $\delta$')
        ax_power_pred.set_xlabel(r'$k$ [$h \ \mathrm{Mpc}^{-1}$]')
        ax_power_pred.set_ylabel(r'$P(k)$ [$h^{-3} \ \mathrm{Mpc}^3$]')

        ax_cdf_pred.set_title(r'cdf of normalized predicted $\rho$')
        ax_cdf_pred.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax_cdf.set_title(r'cdf of normalized $\rho$')
    ax_cdf.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # subfigs[0].tight_layout(pad=5.0, w_pad=2.0, h_pad=2.0)

    plt.savefig(ouput_file)