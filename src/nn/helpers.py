import equinox as eqx
import jax.tree_util as jtu
import jax
import json
from typing import Callable, Tuple
import numpy as np
import nn
from config import Config

def count_parameters(model: eqx.Module):
    return sum(p.size for p in jtu.tree_leaves(eqx.filter(model, eqx.is_array)))

def custom_serializer(obj):
    if isinstance(obj, jax.Array):
        return np.array(obj).tolist() 
    raise TypeError(f"Object of type '{obj.__class__.__name__}' is not JSON serializable")

def save_sequential_model(
        filename : str, 
        config : dict,
        training_stats : dict,
        hyperparams : dict,
        model : eqx.Module):
    
    with open(filename, "wb") as f:
        config_str = json.dumps(config)
        f.write((config_str + "\n").encode())
        training_params_stream = json.dumps(training_stats, default=custom_serializer)
        f.write((training_params_stream + "\n").encode())
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)

def load_sequential_model(
        filename : str, 
        activation : Callable) -> Tuple[eqx.Module, dict, dict]:
    
    with open(filename, "rb") as f:
        config = Config(**json.loads(f.readline().decode()))
        training_stats = json.loads(f.readline().decode())
        hyperparams = json.loads(f.readline().decode())
        model = nn.SequentialModel(
                constructor = nn.UNet if config.model_type == "UNet" else nn.FNO,
                sequence_length = config.steps,
                key=jax.random.PRNGKey(0), 
                parameters = hyperparams, 
                activation = activation)

        return eqx.tree_deserialise_leaves(f, model), config, training_stats