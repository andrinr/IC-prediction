import jax.numpy as jnp
import jax
import optax
import equinox as eqx
from functools import partial
from typing import Callable

@partial(jax.jit, static_argnums=[3, 5, 6])
def update_fn(
        start : jnp.ndarray,
        end : jnp.ndarray,
        model_params,
        model_static,
        optimizer_state : optax.OptState,
        optimizer_static,
        loss_function : Callable):

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
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_array))

    model_params, model_static = eqx.partition(model, eqx.is_array)

    for epoch in range(n_epochs):
        losses = []
        for i, data in enumerate(data_iterator):
            start_d = jax.device_put(data['start'], jax.devices('gpu')[0])
            end_d = jax.device_put(data['end'], jax.devices('gpu')[0])

            model_params, optimizer_state, loss = update_fn(
                start_d,
                end_d,
                model_params,
                model_static,
                optimizer_state,
                optimizer,
                loss_function)
            
            losses.append(loss)
        
        print(f"epoch {epoch}, loss {loss}")
        losses = jnp.array(losses)
        print(losses.mean())

    return model, losses