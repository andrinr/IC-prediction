import jax.numpy as jnp
import jax
import optax
import equinox as eqx
from functools import partial
import time
from data import normalize_inv
from cosmos import PowerSpectrum, compute_overdensity

# @partial(jax.jit)
# def mse(prediction : jax.Array, truth : jax.Array):
#     return jnp.mean((jax.nn.sigmoid(prediction) - jax.nn.sigmoid(truth))**2)

@partial(jax.jit)
def mse(prediction : jax.Array, truth : jax.Array):
    return jnp.mean((prediction - truth) ** 2)

@partial(jax.jit)
def mass_conservation_loss(prediction : jax.Array, truth : jax.Array):
    # [Batch, Frames, Channels, Depth, Height, Width]
    pred_mass = jnp.sum(prediction, axis=[3, 4, 5])
    true_mass = jnp.sum(truth, axis=[3, 4, 5])
    return jnp.mean((pred_mass - true_mass)**2)

@partial(jax.jit)
def power_spectrum_loss(prediction : jax.Array, truth : jax.Array):
    power_spectrum = PowerSpectrum(
        64, 20)
    print(prediction.shape)
    p_pred, k = power_spectrum(prediction)
    p_true, k = power_spectrum(truth)

    return mse(jnp.log(p_pred), jnp.log(p_true))

@partial(jax.jit, static_argnums=[3])
def total_loss(
    prediction : jax.Array, 
    truth : jax.Array,
    attributes : jax.Array,
    single_state_loss : bool):

    # truth / prediction: [Batch, Frames, Channels, Depth, Height, Width]
    # attributes: [Batch, Frames, 2] where 0 min 1 max

    # b, f, c, d, w, h = prediction.shape
    # power_spectrum = PowerSpectrum(
    #     d, 20)

    # normalize_map = jax.vmap(jax.vmap(normalize_inv))
    # power_map = jax.vmap(jax.vmap(power_spectrum))

    # rho_truth = normalize_map(truth, attributes[:, :, 0], attributes[:, :, 1])
    # rho_pred = normalize_map(prediction, attributes[:, :, 0], attributes[:, :, 1])

    # delta_truth, mean = compute_overdensity(rho_truth[0])
    # delta_pred, mean = compute_overdensity(rho_pred[0])

    # # mass_loss = mass_conservation_loss(rho_pred, rho_truth)
    # p_truth, k = power_map(delta_truth)
    # p_pred, k = power_map(delta_pred)

    # power_loss = mse(jnp.log(p_truth), jnp.log(p_pred))

    if single_state_loss:
        mse_loss = mse(truth[:, -1], prediction[:, -1])
    else:
        mse_loss = mse(truth, prediction)

    print(mse_loss)
    #print(power_loss)

    return  mse_loss #+ power_loss

@partial(jax.jit, static_argnums=[1, 4, 5])
def prediction_loss(
        model_params : list,
        model_static : eqx.Module,
        sequence : jax.Array,
        attributes : jax.Array,
        sequential_mode : bool,
        single_state_loss : bool):
    
    """
    Prediction and MSE error. 

    shape of sequence:
    [Batch, Frames, Channels, Depth, Height, Width]
    """
    model = eqx.combine(model_params, model_static)
    model_fn = lambda x : model(x, sequential_mode)
    pred = jax.vmap(model_fn)(sequence)

    return total_loss(pred, sequence[:, 1:], attributes[:, 1:], single_state_loss)

@partial(jax.jit, static_argnums=[3, 4, 5])
def get_batch_loss(
        sequence : jax.Array,
        attributes : jax.Array,
        model_params,
        model_static : eqx.Module,
        sequential_mode : bool,
        single_state_loss : bool):
    
    loss = prediction_loss(
        model_params,
        model_static,
        sequence,
        attributes,
        sequential_mode,
        single_state_loss)

    return loss

@partial(jax.jit, static_argnums=[3, 5, 6, 7])
def learn_batch(
        sequence : jax.Array,
        attributes : jax.Array,
        model_params,
        model_static : eqx.Module,
        optimizer_state : optax.OptState,
        optimizer_static,
        sequential_mode : bool,
        single_state_loss : bool):
    """
    Learn model on batch of sequences.

    shape of sequence:
    [Batch, Frames, Channels, Depth, Height, Width]
    """

    value_and_grad = eqx.filter_value_and_grad(prediction_loss, has_aux=False)

    loss, grad = value_and_grad(
        model_params,
        model_static, 
        sequence,
        attributes,
        sequential_mode,
        single_state_loss)
    
    updates, optimizer_state = optimizer_static.update(grad, optimizer_state)

    model = eqx.combine(model_params, model_static)
    model = eqx.apply_updates(model, updates)
    model_params, model_static = eqx.partition(model, eqx.is_array)

    return model_params, optimizer_state, loss

def train_model(
        model_params,
        model_static : eqx.Module,
        train_data_iterator,
        val_data_iterator,
        learning_rate : float,
        n_epochs : int,
        sequential_mode : bool,
        single_state_loss : bool):
    
    optimizer = optax.adam(learning_rate)
    optimizer_state = optimizer.init(model_params)

    training_loss = []
    validation_loss = []
    timestamps = []

    start = time.time()
    for epoch in range(n_epochs):
        epoch_train_loss = []
        epoch_val_loss = []

        for _, data in enumerate(train_data_iterator):
            data_d = jax.device_put(data['data'], jax.devices('gpu')[0])
            attributes_d = jax.device_put(data['attributes'], jax.devices('gpu')[0])
            model_params, optimizer_state, loss = learn_batch(
                sequence = data_d,
                attributes = attributes_d,
                model_params = model_params,
                model_static = model_static,
                optimizer_state = optimizer_state,   
                optimizer_static = optimizer,
                sequential_mode = sequential_mode,
                single_state_loss = single_state_loss)
            epoch_train_loss.append(loss)

        for _, data in enumerate(val_data_iterator):
            data_d = jax.device_put(data['data'], jax.devices('gpu')[0])
            attributes_d = jax.device_put(data['attributes'], jax.devices('gpu')[0])
            loss = get_batch_loss(
                sequence = data_d,
                attributes = attributes_d,
                model_params = model_params,
                model_static= model_static,
                sequential_mode = sequential_mode,
                single_state_loss = single_state_loss)
            epoch_val_loss.append(loss)
        
        epoch_train_loss = jnp.array(epoch_train_loss)
        epoch_val_loss = jnp.array(epoch_val_loss)
        print(f"epoch {epoch}, train loss {epoch_train_loss.mean()}")
        print(f"epoch {epoch}, val loss {epoch_val_loss.mean()}")
        
        training_loss.append(epoch_train_loss.mean())
        validation_loss.append(epoch_val_loss.mean())
        timestamps.append(time.time() - start)

    return model_params, jnp.array(training_loss), jnp.array(validation_loss), jnp.array(timestamps)