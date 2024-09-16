import jax.numpy as jnp
import jax
import optax
import equinox as eqx
from functools import partial
    
@partial(jax.jit, static_argnums=1)
def mse_loss(
        model_params,
        model_static : eqx.Module,
        state : jax.Array,
        next_state : jax.Array):
    
    model = eqx.combine(model_params, model_static)
    next_pred = jax.vmap(model)(state)
    mse = jnp.mean((next_state - next_pred) ** 2)

    return mse, next_pred

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
    value_and_grad = eqx.filter_value_and_grad(mse_loss, has_aux=True)

    for i in range(n_frames-1):
        (loss, pred), grad = value_and_grad(
            model_params, 
            model_static, 
            sequence[:, i],
            sequence[:, i+1])
        
        updates, optimizer_state = optimizer_static.update(grad, optimizer_state)

        model = eqx.combine(model_params, model_static)
        model = eqx.apply_updates(model, updates)
        model_params, model_static = eqx.partition(model, eqx.is_array)
        # model_params = optax.apply_updates(model_params, updates)

    return model_params, optimizer_state, loss

def train_model(
        model_params,
        model_static : eqx.Module,
        train_data_iterator,
        val_data_iterator,
        learning_rate : float,
        n_epochs : int):
    
    optimizer = optax.adam(learning_rate)
    optimizer_state = optimizer.init(model_params)

    training_loss = []
    validation_loss = []

    for epoch in range(n_epochs):
        epoch_train_loss = []
        epoch_val_loss = []

        for _, data in enumerate(train_data_iterator):
            sequence_d = jax.device_put(data['sequence'], jax.devices('gpu')[0])

            model_params, optimizer_state, loss = learn_batch(
                sequence_d,
                model_params,
                model_static,
                optimizer_state,
                optimizer)
            epoch_train_loss.append(loss)

        for _, data in enumerate(val_data_iterator):
            sequence_d = jax.device_put(data['sequence'], jax.devices('gpu')[0])

            model_params, optimizer_state, loss = learn_batch(
                sequence_d,
                model_params,
                model_static,
                optimizer_state,
                optimizer)
            epoch_val_loss.append(loss)
        
        epoch_train_loss = jnp.array(epoch_train_loss)
        epoch_val_loss = jnp.array(epoch_val_loss)
        print(f"epoch {epoch}, train loss {epoch_train_loss.mean()}")
        print(f"epoch {epoch}, val loss {epoch_val_loss.mean()}")
        training_loss.append(epoch_train_loss.mean())
        validation_loss.append(epoch_val_loss.mean())

    return model_params, jnp.array(training_loss), jnp.array(validation_loss)