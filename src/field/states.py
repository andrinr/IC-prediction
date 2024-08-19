import jax
from typing import NamedTuple

class Field(NamedTuple):
    grid : jax.Array
    dx : float
