import jax.numpy as jnp
import jax
import optax
import equinox as eqx
from functools import partial

@partial(jax.jit, static_argnums=1)
def loss_fn(params, static, x : jnp.ndarray, y : jnp.ndarray):
    model = eqx.combine(params, static)
    y_pred = jax.vmap(model)(x)
    mse = jnp.mean(jnp.square(y_pred - y))
    return mse

@eqx.filter_jit
def update_fn(
        start : jnp.ndarray,
        end : jnp.ndarray,
        model : eqx.Module, 
        optimizer_state : optax.OptState,
        optimizer):

    params, static = eqx.partition(model, eqx.is_array)

    loss, grad = eqx.filter_value_and_grad(loss_fn)(
        params, static, end, start)
    updates, new_optimizer_state = optimizer.update(grad, optimizer_state)
    new_model = eqx.apply_updates(model, updates)

    return new_model, new_optimizer_state, loss

def train(
        model,
        data_iterator,
        learning_rate : float,
        n_epochs : int):
    
    optimizer = optax.adam(learning_rate)
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_array))

    for _ in range(n_epochs):
        losses = []
        for i, data in enumerate(data_iterator):
            start_d = jax.device_put(data['start'], jax.devices('gpu')[0])
            end_d = jax.device_put(data['end'], jax.devices('gpu')[0])
            
            model, optimizer_state, loss = update_fn(
                start_d,
                end_d,
                model,  
                optimizer_state,
                optimizer)
            
            losses.append(loss)
        
        losses = jnp.array(losses)
        print(losses.mean())

    return model, losses