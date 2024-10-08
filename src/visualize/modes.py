import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from config import Config
from mpl_toolkits.axes_grid1 import make_axes_locatable
from powerbox import get_power
from cosmos import compute_overdensity, to_redshift, normalize, normalize_inv
from matplotlib.cm import get_cmap

def modes(
        output_file: str,
        config: Config,
        sequence: jax.Array,
        predictions: jax.Array,
        attributes: jax.Array,
        norm_functions : str):
    
    num_predictions = len(predictions)

    print(num_predictions)

    # Create figure
    fig = plt.figure(layout='constrained', figsize=(10, 4),  constrained_layout=True)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    
    frames = sequence.shape[0]
    grid_size = sequence.shape[2]
        
    sequence_curr = jnp.reshape(sequence, (frames, grid_size, grid_size, grid_size, 1))
    attributes_curr = attributes[-1]
    attribs = jax.device_put(attributes_curr, device=jax.devices("gpu")[0])
    normalized = sequence_curr[-1]
    rho = normalize_inv(normalized, attribs, norm_functions)
    delta = compute_overdensity(rho)

    pred_curr = jnp.reshape(predictions, (frames - 1, grid_size, grid_size, grid_size, 1))
    normalized_pred = pred_curr[-1]
    rho_pred = normalize_inv(normalized_pred, attribs, norm_functions)
    delta_pred = compute_overdensity(rho_pred)

    N = normalized.shape[1]
    norm_fs = jnp.fft.rfftn(normalized, s=(N, N, N), axes=(0, 1, 2))
    norm_pred_fs = jnp.fft.rfftn(normalized_pred, s=(N, N, N), axes=(0, 1, 2))
    kx = jnp.fft.fftfreq(N)[:, None, None]
    ky = jnp.fft.fftfreq(N)[None, :, None]
    kz = jnp.fft.rfftfreq(N)[None, None, :]

    k_squared = kx**2 + ky**2 + kz**2
    cutoff_k_squared = 0.01

    # Mask out higher wavelengths
    mask = (k_squared <= cutoff_k_squared)[:, :, :, None]
    # mask = k_squared <= cutoff_k_squared
    norm_fs_filtered = norm_fs * mask
    norm_pred_fs_filtered = norm_pred_fs * mask

    # Transform back to real space
    norm_filtered = jnp.fft.irfftn(norm_fs_filtered, s=(N, N, N), axes=(0, 1, 2))
    norm_pred_filtered = jnp.fft.irfftn(norm_pred_fs_filtered, s=(N, N, N), axes=(0, 1, 2))
    defect = norm_filtered - norm_pred_filtered

    ax1.set_title(r'$\rho_{norm} - \hat{\rho}_{norm}$')
    # ax_seq.set_title(r'input $\rho$')
    ax1.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    im_seq = ax1.imshow(defect[N//2], cmap='RdYlBu')
    # im_seq = ax_seq.imshow(norm_filtered[grid_size // 2, :, :], cmap='inferno')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('bottom', size='5%', pad=0.03)
    fig.colorbar(im_seq, cax=cax, orientation='horizontal')
    

    ax2.set_title(r'$\rho_{norm}$')
    ax2.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    im_seq = ax2.imshow(norm_pred_filtered[N//2], cmap='inferno')
    # im_seq = ax_seq.imshow(norm_filtered[grid_size // 2, :, :], cmap='inferno')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('bottom', size='5%', pad=0.03)
    fig.colorbar(im_seq, cax=cax, orientation='horizontal')

    ax3.set_title(r'$\hat{\rho}_{norm}$')
    ax3.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    im_seq = ax3.imshow(norm_filtered[N//2], cmap='inferno')
    # im_seq = ax_seq.imshow(norm_filtered[grid_size // 2, :, :], cmap='inferno')
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('bottom', size='5%', pad=0.03)
    fig.colorbar(im_seq, cax=cax, orientation='horizontal')

    plt.savefig("img/low_modes.jpg")

    # Create figure
    fig = plt.figure(layout='constrained', figsize=(10, 4),  constrained_layout=True)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    # Mask out lower wavelengths
    mask = (k_squared >= cutoff_k_squared)[:, :, :, None]
    # mask = k_squared <= cutoff_k_squared
    norm_fs_filtered = norm_fs * mask
    norm_pred_fs_filtered = norm_pred_fs * mask
    norm_filtered = jnp.fft.irfftn(norm_fs_filtered, s=(N, N, N), axes=(0, 1, 2))
    norm_pred_filtered = jnp.fft.irfftn(norm_pred_fs_filtered, s=(N, N, N), axes=(0, 1, 2))

    ax1.hist(
        norm_filtered.flatten(),
        100, 
        density=True, 
        log=True, 
        histtype="step",
        cumulative=False, 
        label=fr'simulation')

    ax1.hist(
        norm_pred_filtered.flatten(),
        100, 
        density=True, 
        log=True, 
        histtype="step",
        cumulative=False, 
        label=fr'prediction')
    ax1.legend()
    # plot cdf 

    ax2.set_title(r'$\rho_{norm}$')
    ax2.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    im_seq = ax2.imshow(norm_filtered[N//2], cmap='inferno')
    # im_seq = ax_seq.imshow(norm_filtered[grid_size // 2, :, :], cmap='inferno')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('bottom', size='5%', pad=0.03)
    fig.colorbar(im_seq, cax=cax, orientation='horizontal')

    ax3.set_title(r'$\hat{\rho}_{norm}$')
    ax3.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    im_seq = ax3.imshow(norm_pred_filtered[N//2], cmap='inferno')
    # im_seq = ax_seq.imshow(norm_filtered[grid_size // 2, :, :], cmap='inferno')
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('bottom', size='5%', pad=0.03)
    fig.colorbar(im_seq, cax=cax, orientation='horizontal')


    plt.savefig("img/high_modes.jpg")