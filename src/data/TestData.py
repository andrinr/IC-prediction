# from __future__ import annotations, type
import jax.numpy as jnp
from random import shuffle
import os
import nvidia.dali.types as types
import random

type Range = tuple[str, int]

class TestData:
    def __init__(
            self, 
            batch_size : int, 
            grid_size : int, 
            range : Range):

        self.dir = os.path.abspath(directory)
        self.batch_size = batch_size
        self.grid_size = grid_size

        self.keys = jax.random

    def __call__(self, sample_info : types.SampleInfo):
        
        sequence = []
        

        return sequence
