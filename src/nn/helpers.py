import equinox as eqx
import jax.tree_util as jtu
import jax
import json
from typing import Callable, Tuple
import numpy as np
from config import Config
from .metric import Metric
from .sequential_model import SequentialModel
from .fno import FNO
from .unet import UNet

def count_parameters(model: eqx.Module):
    return sum(p.size for p in jtu.tree_leaves(eqx.filter(model, eqx.is_array)))


def custom_serializer(obj):
    if isinstance(obj, jax.Array):
        # Convert JAX array to a list
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
        train_stats_str = json.dumps(training_stats, default=custom_serializer)
        f.write((train_stats_str + "\n").encode())
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)

def load_sequential_model(
        filename : str) -> Tuple[eqx.Module, Config, dict]:
    
    with open(filename, "rb") as f:
        config = Config(**json.loads(f.readline().decode()))
        training_stats = json.loads(f.readline().decode())
        hyperparams = json.loads(f.readline().decode())
        model = SequentialModel(
                constructor = UNet if config.model_type == "UNet" else FNO,
                sequence_length = config.file_index_steps,
                unique_networks = config.unique_networks,
                key=jax.random.PRNGKey(0), 
                sequential_skip_channels=config.sequential_skip_channels,
                parameters = hyperparams)

        return eqx.tree_deserialise_leaves(f, model), config, training_stats