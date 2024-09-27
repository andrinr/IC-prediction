import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from config import Config
from cosmos import PowerSpectrum
from mpl_toolkits.axes_grid1 import make_axes_locatable
from powerbox import PowerBox, get_power
from data import normalize_inv
from cosmos import compute_overdensity

def sequence_examine(
        ouput_file : str,
        config : Config,
        sequence : jax.Array,
        timeline : jax.Array,
        attributes : jax.Array):
    
    frames = sequence.shape[0]
    grid_size = sequence.shape[2]

    # transform to shape for matplotlib
    sequence = jnp.reshape(
        sequence, (frames, grid_size, grid_size, grid_size, 1))


    fig = plt.figure(figsize=(4+3*frames, 8), layout="constrained")
    spec = fig.add_gridspec(2, frames)

    ax_cdf = fig.add_subplot(spec[1, 0:2])
    ax_power = fig.add_subplot(spec[1, 2:4])

    for frame in range(frames):

        min = attributes[frame, 0]
        max = attributes[frame, 1]

        min = jax.device_put(min, device=jax.devices("gpu")[0])
        max = jax.device_put(max, device=jax.devices("gpu")[0])

        rho_normalized = sequence[frame]
        rho = normalize_inv(rho_normalized, min, max)

        ax_seq = fig.add_subplot(spec[0, frame])
        ax_seq.set_title(f"sim t: {timeline[frame]}")

        im_seq = ax_seq.imshow(rho_normalized[grid_size // 2, : , :], cmap='inferno')
        fig.colorbar(im_seq, ax=ax_seq, orientation='horizontal', location='bottom')

        ax_cdf.hist(rho_normalized.flatten(), 100, density=True, log=True, histtype="step",
                               cumulative=False, label=f"sim t: {timeline[frame]}")
        
        delta, mean = compute_overdensity(rho)

        print(min)
        print(max)
        
        p,k = get_power(delta[:, :, :, 0], config.box_size)
        print(p)
        print(k)
        ax_power.plot(k, p, label=f"sim t: {timeline[frame]}")

    ax_power.set_yscale('log')
    ax_power.set_xscale('log')
    ax_power.legend()
    ax_power.set_xlabel(r'$k$ [$h \ \mathrm{Mpc}^{-1}$]')
    ax_power.set_ylabel(r'$P(k)$ [$h^{-3} \ \mathrm{Mpc}^3$]')

    ax_cdf.set_title(r'cdf of $\log_{10} \rho$')
    # ax_cdf_rho.set_title(r'cdf of $\rho$')

    ax_cdf.legend()
    # ax_cdf.set_xlim(0, 1)
    
    plt.savefig(ouput_file)

def sequence_examine_prediction(
        ouput_file : str,
        config : Config,
        sequence : jax.Array,
        sequence_prediction : jax.Array,
        timeline : jax.Array,
        attributes : jax.Array):
    
    frames = sequence.shape[0]
    grid_size = sequence.shape[2]

    # transform to shape for matplotlib
    sequence = jnp.reshape(
        sequence, (frames, grid_size, grid_size, grid_size, 1))
    sequence_prediction = jnp.reshape(
        sequence_prediction, (frames-1, grid_size, grid_size, grid_size, 1))

    fig = plt.figure(figsize=(4+3*frames, 8), layout="constrained")
    spec = fig.add_gridspec(3, frames)

    ax_cdf = fig.add_subplot(spec[2, 0])
    ax_power = fig.add_subplot(spec[2, 1])

    for frame in range(frames):

        min = attributes[frame, 0]
        max = attributes[frame, 1]

        min = jax.device_put(min, device=jax.devices("gpu")[0])
        max = jax.device_put(max, device=jax.devices("gpu")[0])

        rho_normalized = sequence[frame]
        rho = normalize_inv(rho_normalized, min, max)
        delta, mean = compute_overdensity(rho)

        ax_seq = fig.add_subplot(spec[0, frame])
        ax_seq.set_title(f"sim t: {timeline[frame]}")

        im_seq = ax_seq.imshow(rho_normalized[grid_size // 2, : , :], cmap='inferno')
        fig.colorbar(im_seq, ax=ax_seq, orientation='horizontal', location='bottom')

        if frame < frames-1:
            rho_pred_normalized = sequence_prediction[frame]
            rho_pred = normalize_inv(rho_pred_normalized, min, max)
            delta_pred, mean_pred = compute_overdensity(rho_pred)

            ax_seq = fig.add_subplot(spec[1, frame+1])
            ax_seq.set_title(f"pred t: {timeline[frame+1]}")

            im_seq = ax_seq.imshow(rho_pred_normalized[grid_size // 2, : , :], cmap='inferno')
            fig.colorbar(im_seq, ax=ax_seq, orientation='horizontal', location='bottom')

            ax_cdf.hist(rho_pred_normalized.flatten(), 100, density=True, log=True, histtype="step",
                               cumulative=False, label=f"pred t: {timeline[frame+1]}")
            p,k = get_power(delta_pred[:, :, :, 0], config.box_size)
            ax_power.plot(k, p, label=f"pred t: {timeline[frame+1]}")


        ax_cdf.hist(rho_normalized.flatten(), 100, density=True, log=True, histtype="step",
                               cumulative=False, label=f"sim t: {timeline[frame]}")
        p,k = get_power(delta[:, :, :, 0], config.box_size)
        ax_power.plot(k, p, label=f"sim t: {timeline[frame]}")

    ax_power.set_yscale('log')
    ax_power.set_xscale('log')
    ax_power.legend()
    ax_power.set_xlabel(r'$k$ [$h \ \mathrm{Mpc}^{-1}$]')
    ax_power.set_ylabel(r'$P(k)$ [$h^{-3} \ \mathrm{Mpc}^3$]')

    ax_cdf.set_title(r'cdf of $\log_{10} \rho$')
    # ax_cdf_rho.set_title(r'cdf of $\rho$')

    ax_cdf.legend()
    # ax_cdf.set_xlim(0, 1)
    
    plt.savefig(ouput_file)