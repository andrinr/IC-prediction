import jax.numpy as jnp
import jax
import optax
import equinox as eqx
from functools import partial
    
@partial(jax.jit, static_argnums=3)
def mse_loss(
        model_params,
        model_static : eqx.Module,
        state : jax.Array,
        next_state : jax.Array):
    
    model = eqx.combine(model_params, model_static)
    next_state_pred = jax.vmap(model)(state)
    mse = jnp.mean((next_state - next_state_pred) ** 2)

    return mse, next_state_pred

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
    total_grad = []
    value_and_grad = eqx.filter_value_and_grad(mse_loss)

    for i in range(n_frames-1):
        loss, grad = value_and_grad(
            model_params, 
            model_static, 
            sequence[i],
            sequence[i+1])
        
    updates, new_optimizer_state = optimizer_static.update(total_grad, optimizer_state)

    new_params = optax.apply_updates(model_params, updates)

    return new_params, new_optimizer_state, loss

def train_model(
        model_params,
        model_static : eqx.Module,
        data_iterator,
        learning_rate : float,
        n_epochs : int):
    
    optimizer = optax.adam(learning_rate)
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

    return model_params, jnp.array(epoch_loss)