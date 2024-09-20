import jax.numpy as jnp
import jax
import optax
import equinox as eqx
from functools import partial
import time

@partial(jax.jit, static_argnums=[1, 3])
def mse_loss(
        model_params : list,
        model_static : eqx.Module,
        sequence : jax.Array,
        sequential_mode = bool):
    
    """
    Prediction and MSE error. 

    shape of sequence:
    [Batch, Frames, Channels, Depth, Height, Width]
    """
    model = eqx.combine(model_params, model_static)
    model_fn = lambda x : model(x, sequential_mode)
    pred = jax.vmap(model_fn)(sequence)
    mse = jnp.mean((pred[1:] - sequence[1:]) ** 2)

    return mse

@partial(jax.jit, static_argnums=[2, 3])
def get_batch_loss(
        sequence : jax.Array,
        model_params,
        model_static : eqx.Module,
        sequential_mode : bool):
    
    loss = mse_loss(
        model_params,
        model_static, 
        sequence,
        sequential_mode)
    
    return loss

@partial(jax.jit, static_argnums=[2, 4, 5])
def learn_batch(
        sequence : jax.Array,
        model_params,
        model_static : eqx.Module,
        optimizer_state : optax.OptState,
        optimizer_static,
        sequential_mode = bool):
    """
    Learn model on batch of sequences.

    shape of sequence:
    [Batch, Frames, Channels, Depth, Height, Width]
    """

    value_and_grad = eqx.filter_value_and_grad(mse_loss, has_aux=False)

    loss, grad = value_and_grad(
        model_params,
        model_static, 
        sequence,
        sequential_mode)
    
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
        sequential_mode : bool):
    
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
            model_params, optimizer_state, loss = learn_batch(
                data_d,
                model_params,
                model_static,
                optimizer_state,
                optimizer,
                sequential_mode = sequential_mode)
            epoch_train_loss.append(loss)

        for _, data in enumerate(val_data_iterator):
            data_d = jax.device_put(data['data'], jax.devices('gpu')[0])
            loss = get_batch_loss(
                data_d,
                model_params,
                model_static,
                sequential_mode = sequential_mode)
            epoch_val_loss.append(loss)
        
        epoch_train_loss = jnp.array(epoch_train_loss)
        epoch_val_loss = jnp.array(epoch_val_loss)
        print(f"epoch {epoch}, train loss {epoch_train_loss.mean()}")
        print(f"epoch {epoch}, val loss {epoch_val_loss.mean()}")
        training_loss.append(epoch_train_loss.mean())
        validation_loss.append(epoch_val_loss.mean())
        timestamps.append(time.time() - start)

    return model_params, jnp.array(training_loss), jnp.array(validation_loss), jnp.array(timestamps)