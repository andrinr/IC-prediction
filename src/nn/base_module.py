from typing import Callable
import jax
import equinox as eqx

class BaseModule(eqx.Module):

    activation : callable

    def __init__(
            self, 
            activation : str):
        
        if activation == "relu":
            self.activation = jax.nn.relu
        elif activation == "tanh":
            self.activation = jax.nn.tanh
