import equinox as eqx
import jax.tree_util as jtu
import jax
import json
from typing import Callable
import os

def count_parameters(model: eqx.Module):
    return sum(p.size for p in jtu.tree_leaves(eqx.filter(model, eqx.is_array)))

def save(
        filename : str, 
        hyperparams : dict,
        model : eqx.Module):
    
    file = os.path.abspath(filename)
    with open(file, "wb") as f:
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)

def load(
        filename : str, 
        constructor, 
        activation : Callable):
    
    file = os.path.abspath(filename)

    with open(file, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        model = constructor(
            key=jax.random.PRNGKey(0), **hyperparams, activation=activation)
        return eqx.tree_deserialise_leaves(f, model)
