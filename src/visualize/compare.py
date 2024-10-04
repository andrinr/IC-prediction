import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from config import Config
from mpl_toolkits.axes_grid1 import make_axes_locatable
from powerbox import get_power
from cosmos import compute_overdensity, to_redshift, normalize, normalize_inv
from matplotlib.cm import get_cmap

def compare(
        output_file: str,
        config: Config,
        sequences: list[jax.Array],
        predictions: list[jax.Array],
        labels : list[str],
        attributes: list[jax.Array],
        norm_functions : list[str]):
    
    num_predictions = len(predictions)

    print(num_predictions)

    # Create figure
    fig = plt.figure(layout='constrained', figsize=(4 + 3 * num_predictions, 10),  constrained_layout=True)
    subfigs = fig.subfigures(2, 1, wspace=0.07, hspace=0.1, height_ratios=[2, 1] if num_predictions > 0 else [1, 1])
    spec_sequence = subfigs[0].add_gridspec(2, num_predictions+1, wspace=0.3, hspace=0.1)
    spec_stats = subfigs[1].add_gridspec(1, 1)
    
    # Main sequence analysis part
    # ax_cdf = fig.add_subplot(spec_stats[0], adjustable='box', aspect=0.1)
    ax_power = fig.add_subplot(spec_stats[0], adjustable='box', aspect=0.1)
    cmap = get_cmap('viridis')
    colors = cmap(jnp.linspace(0, 1, 6))
    
    file_index_stride = config.file_index_stride
    step = config.file_index_start

    frames = sequences[0].shape[0]
    grid_size = sequences[0].shape[2]
        
    sequence_curr = jnp.reshape(sequences[0], (frames, grid_size, grid_size, grid_size, 1))
    attributes_curr = attributes[0]
    attribs = jax.device_put(attributes_curr[1], device=jax.devices("gpu")[0])
    normalized = sequence_curr[1]
    rho = normalize_inv(normalized, attribs, norm_functions[0])
    delta = compute_overdensity(rho)

    ax_seq = fig.add_subplot(spec_sequence[1, 0])
    ax_seq.set_title(r'output $\rho_{norm}$')
    ax_seq.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    im_seq = ax_seq.imshow(normalized[grid_size // 2, :, :], cmap='inferno')

    normalized = sequence_curr[0]
    ax_seq = fig.add_subplot(spec_sequence[0, 0])
    ax_seq.set_title(r'input $\rho_{norm}$')
    ax_seq.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    im_seq = ax_seq.imshow(normalized[grid_size // 2, :, :], cmap='inferno')

    p, k = get_power(delta[:, :, :, 0], config.box_size)
    ax_power.plot(
        k,
        p,
        label=fr'sim $z = {to_redshift(step/100):.2f}$')
        # color=colors[frame])
    
    # Process predictions
    for idx in range(num_predictions):

        frames = sequences[idx].shape[0]
        grid_size = sequences[idx].shape[2]
        
        # Transform to shape for matplotlib
        sequence_curr = jnp.reshape(sequences[idx], (frames, grid_size, grid_size, grid_size, 1))
        pred_curr = jnp.reshape(predictions[idx], (frames - 1, grid_size, grid_size, grid_size, 1))
        attributes_curr = attributes[idx]

        print(sequence_curr.shape)
        print(pred_curr.shape)
        print(attributes_curr.shape)

        print("hello")
        # return
        
        frame = 0
        attribs = jax.device_put(attributes_curr[frame+1], device=jax.devices("gpu")[0])
        normalized = sequence_curr[frame+1]
        rho = normalize_inv(normalized, attribs, norm_functions[idx])
        delta = compute_overdensity(rho)

        rho_pred_normalized = pred_curr[frame]
        attribs = jax.device_put(attributes_curr[frame + 1], device=jax.devices("gpu")[0])
        rho_pred = normalize_inv(rho_pred_normalized, attribs,  norm_functions[idx])
        delta_pred = compute_overdensity(rho_pred)

        normalized = sequence_curr[0]
        N = normalized.shape[1]
        # x shape : n_channels, N, N, N
        # x_fs shape : n_channels, N, N, N // 2, 2

        norm_fs = jnp.fft.rfftn(normalized, s=(N, N, N), axes=(0, 1, 2))
        kx = jnp.fft.fftfreq(N)[:, None, None]
        ky = jnp.fft.fftfreq(N)[None, :, None]
        kz = jnp.fft.rfftfreq(N)[None, None, :]

        print(norm_fs.shape)

        k_squared = kx**2 + ky**2 + kz**2
        cutoff_k_squared = 0.02

        # Mask out higher wavelengths
        mask = (k_squared <= cutoff_k_squared)[:, :, :, None]
        # mask = k_squared <= cutoff_k_squared
        norm_fs_filtered = norm_fs * mask

        # Transform back to real space
        norm_filtered = jnp.fft.irfftn(norm_fs_filtered, s=(N, N, N), axes=(0, 1, 2))
        rho_pred_filtered = normalize_inv(norm_filtered, attribs,  norm_functions[idx])
        delta_pred_filtered = compute_overdensity(rho_pred_filtered)

        ax_seq = fig.add_subplot(spec_sequence[0, idx+1])
        ax_seq.set_title(r'$\rho_{norm} - \hat{\rho}_{norm}$')
        # ax_seq.set_title(r'input $\rho$')
        ax_seq.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        im_seq = ax_seq.imshow(normalized[grid_size // 2, :, :] - rho_pred_normalized[grid_size // 2, :, :], cmap='RdYlBu')
        # im_seq = ax_seq.imshow(norm_filtered[grid_size // 2, :, :], cmap='inferno')

        divider = make_axes_locatable(ax_seq)
        cax = divider.append_axes('bottom', size='5%', pad=0.03)
        fig.colorbar(im_seq, cax=cax, orientation='horizontal')
        
        # ax_cdf.hist(
        #     normalized.flatten(),
        #     20,
        #     density=True,
        #     log=True,
        #     histtype="step",
        #     cumulative=False,
        #     label=fr'sim $z = {to_redshift(step/100):.2f}$')
        #     # color=colors[frame])

        # prediction
        step += file_index_stride * (-1 if config.flip else 1)

        ax_seq_pred = fig.add_subplot(spec_sequence[1, idx+1])
        ax_seq_pred.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax_seq_pred.set_title(r'$\hat{\rho}_{norm}$' + f' {labels[idx]}')
        im_seq_pred = ax_seq_pred.imshow(rho_pred_normalized[grid_size // 2, :, :], cmap='inferno')
        
        divider_pred = make_axes_locatable(ax_seq_pred)
        cax_pred = divider_pred.append_axes('bottom', size='5%', pad=0.03)
        fig.colorbar(im_seq_pred, cax=cax_pred, orientation='horizontal')
        
        # ax_cdf.hist(
        #     rho_pred_normalized.flatten(),
        #     100,
        #     density=True,
        #     log=True,
        #     histtype="step",
        #     cumulative=False,
        #     label=fr'pred {labels[idx]} $z = {to_redshift(step/100):.2f}$')
        #     # color=colors[frame + 1])
        
        p_pred, k_pred = get_power(delta_pred[:, :, :, 0], config.box_size)
        ax_power.plot(
            k_pred, p_pred,
            label=fr'pred {labels[idx]}')
        
        # p_pred, k_pred = get_power(delta_pred_filtered[:, :, :, 0], config.box_size)
        # ax_power.plot(
        #     k_pred, p_pred,
        #     label=fr'filtered z={labels[idx]}')
        #     # color=colors[frame + 1])

    # Finalize plots
    
    ax_power.set_yscale('log')
    ax_power.set_xscale('log')
    ax_power.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax_power.set_title(r'Power Spectrum of $\delta$')
    ax_power.set_xlabel(r'$k$ [$h \ \mathrm{Mpc}^{-1}$]')
    ax_power.set_ylabel(r'$P(k)$ [$h^{-3} \ \mathrm{Mpc}^3$]')
    # ax_cdf.set_title(r'pdf $\rho_{norm}$')
    # ax_cdf.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.3, wspace=0.3)
    plt.savefig(output_file)