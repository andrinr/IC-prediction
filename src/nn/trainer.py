import jax.numpy as jnp
import jax
import optax
import equinox as eqx
from functools import partial
import time

# @partial(jax.jit)
# def mse(prediction : jax.Array, truth : jax.Array):
#     return jnp.mean((jax.nn.sigmoid(prediction) - jax.nn.sigmoid(truth))**2)

@partial(jax.jit)
def mse(prediction : jax.Array, truth : jax.Array):
    return jnp.mean((prediction - truth) ** 2)

@partial(jax.jit)
def baseline_loss(
        sequence : jax.Array):
    
    """
    Prediction and MSE error. 

    shape of sequence:
    [Batch, Frames, Channels, Depth, Height, Width]
    """
    a = sequence[:, :-1]
    # a_mean = a.mean()
    # a_var = a.var()
    b = sequence[:, 1:]
    # b_mean = b.mean()
    # b_var= b.var()

    # normalize a
    # a = (a - a_mean) / a_var
    # fit it to b distribution
    # a = (a * b_var) + b_mean

    return mse(b, a)

@partial(jax.jit, static_argnums=[1, 3])
def prediction_loss(
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

    return mse(pred, sequence[:, 1:])

@partial(jax.jit, static_argnums=[2, 3])
def get_batch_loss(
        sequence : jax.Array,
        model_params,
        model_static : eqx.Module,
        sequential_mode : bool):
    
    loss = prediction_loss(
        model_params,
        model_static, 
        sequence,
        sequential_mode)
    
    baseline = baseline_loss(sequence)
    
    return loss, baseline

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

    value_and_grad = eqx.filter_value_and_grad(prediction_loss, has_aux=False)

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
    baseline_loss = []
    timestamps = []

    start = time.time()
    for epoch in range(n_epochs):
        epoch_train_loss = []
        epoch_val_loss = []
        epoch_baseline_loss = []

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
            loss, baseline = get_batch_loss(
                data_d,
                model_params,
                model_static,
                sequential_mode = sequential_mode)
            epoch_val_loss.append(loss)
            epoch_baseline_loss.append(baseline)
        
        epoch_train_loss = jnp.array(epoch_train_loss)
        epoch_val_loss = jnp.array(epoch_val_loss)
        epoch_baseline_loss = jnp.array(epoch_baseline_loss)
        print(f"epoch {epoch}, train loss {epoch_train_loss.mean()}")
        print(f"epoch {epoch}, val loss {epoch_val_loss.mean()}")
        print(f"epoch {epoch}, baseline loss {epoch_baseline_loss.mean()}")
        
        training_loss.append(epoch_train_loss.mean())
        validation_loss.append(epoch_val_loss.mean())
        baseline_loss.append(epoch_baseline_loss.mean())
        timestamps.append(time.time() - start)

    return model_params, jnp.array(training_loss), jnp.array(validation_loss), jnp.array(baseline_loss), jnp.array(timestamps)