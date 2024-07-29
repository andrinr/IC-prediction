import equinox as eqx
import jax.tree_util as jtu
import jax
import json

def count_parameters(model: eqx.Module):
    return sum(p.size for p in jtu.tree_leaves(eqx.filter(model, eqx.is_array)))

def save(filename, hyperparams, model):
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)

def load(filename, model_constructor):
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        model = model_constructor(key=jax.random.PRNGKey(0), **hyperparams)
        return eqx.tree_deserialise_leaves(f, model)
