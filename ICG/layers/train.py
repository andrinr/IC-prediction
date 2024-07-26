import jax.numpy as jnp
import jax
import optax
import equinox as eqx
from functools import partial
from typing import Callable

@partial(jax.jit, static_argnums=[3, 5, 6])
def learn_batch(
        start : jax.Array,
        end : jax.Array,
        model_params,
        model_static,
        optimizer_state : optax.OptState,
        optimizer_static,
        loss_function : Callable):
    """
    Learn model on a data batch.
    """

    loss, grad = eqx.filter_value_and_grad(loss_function)(
        model_params, model_static, end, start)
    
    updates, new_optimizer_state = optimizer_static.update(grad, optimizer_state)

    model =  eqx.combine(model_params, model_static)
    new_model = eqx.apply_updates(model, updates)
    new_params, _ = eqx.partition(new_model, eqx.is_array)

    return new_params, new_optimizer_state, loss

def train(
        model,
        data_iterator,
        learning_rate : float,
        n_epochs : int,
        loss_function : Callable):
    
    optimizer = optax.adam(learning_rate)
    model_params, model_static = eqx.partition(model, eqx.is_array)
    optimizer_state = optimizer.init(model_params)

    training_loss = []
    for epoch in range(n_epochs):
        epoch_loss = []
        for _, data in enumerate(data_iterator):
            start_d = jax.device_put(data['start'], jax.devices('gpu')[0])
            end_d = jax.device_put(data['end'], jax.devices('gpu')[0])

            model_params, optimizer_state, loss = learn_batch(
                start_d,
                end_d,
                model_params,
                model_static,
                optimizer_state,
                optimizer,
                loss_function)
            epoch_loss.append(loss)
        
        epoch_loss = jnp.array(epoch_loss)
        print(f"epoch {epoch}, loss {epoch_loss.mean()}")
        training_loss.append(epoch_loss.mean())

    model = eqx.combine(model_params, model_static)

    return model, jnp.array(epoch_loss)