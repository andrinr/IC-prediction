import jax.numpy as jnp
import jax
import optax
import equinox as eqx
from functools import partial
from typing import Callable

# "always scan when you can!"
@partial(jax.jit, static_argnums=[1, 3])
def predict_sequence(
        init : jax.Array,
        steps : int,
        model_params,
        model_static : eqx.Module):
    
    model = eqx.combine(model_params, model_static)

    def scan_fn(state, _):
        new_state = model(state)
        return new_state, new_state
    
    _, traj = jax.lax.scan(
        f = scan_fn, init=init, xs=None, length=steps)
    
    return traj

@partial(jax.jit, static_argnums=[1, 2])
def mse_sequence_loss(
        model_params,
        model_static : eqx.Module,
        steps : int,
        sequence : jax.Array):
    """
    Compute loss on batch of sequences.

    shape of sequence:
    [Batch, Frames, Channels, Depth, Height, Width]
    """
    
    init_state = sequence.at[:, 0].get()

    predict = lambda x : predict_sequence(x, steps, model_params, model_static)

    # map sequence prediction over all batches
    predicted_sequence = jax.vmap(predict)(init_state)
    
    mse = jnp.mean((sequence[:, 1:] - predicted_sequence) ** 2)
    return mse

@partial(jax.jit, static_argnums=[2, 4])
def learn_batch(
        sequence : jax.Array,
        model_params,
        model_static : eqx.Module,
        optimizer_state : optax.OptState,
        optimizer_static):
    """
    Learn model on batch of sequences.

    shape of sequence:
    [Batch, Frames, Channels, Depth, Height, Width]
    """

    print(sequence.shape)

    n_frames = sequence.shape[1]
    value_and_grad = eqx.filter_value_and_grad(mse_sequence_loss)

    loss, grad = value_and_grad(
        model_params, 
        model_static, 
        n_frames - 1,
        sequence)
    
    updates, new_optimizer_state = optimizer_static.update(grad, optimizer_state)

    model =  eqx.combine(model_params, model_static)
    new_model = eqx.apply_updates(model, updates)
    new_params, _ = eqx.partition(new_model, eqx.is_array)

    return new_params, new_optimizer_state, loss

def train_model(
        model,
        data_iterator,
        learning_rate : float,
        n_epochs : int):
    
    optimizer = optax.adam(learning_rate)
    model_params, model_static = eqx.partition(model, eqx.is_array)
    optimizer_state = optimizer.init(model_params)

    training_loss = []
    for epoch in range(n_epochs):
        epoch_loss = []
        for _, data in enumerate(data_iterator):
            sequence_d = jax.device_put(data['sequence'], jax.devices('gpu')[0])

            model_params, optimizer_state, loss = learn_batch(
                sequence_d,
                model_params,
                model_static,
                optimizer_state,
                optimizer)
            epoch_loss.append(loss)
        
        epoch_loss = jnp.array(epoch_loss)
        print(f"epoch {epoch}, loss {epoch_loss.mean()}")
        training_loss.append(epoch_loss.mean())

    model = eqx.combine(model_params, model_static)

    return model, jnp.array(epoch_loss)