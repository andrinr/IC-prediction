# from __future__ import annotations, type
import jax.numpy as jnp
from random import shuffle
import os
import nvidia.dali.types as types
import random

type Range = tuple[str, int]

class VolumetricSequence:
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
        stride = 1 if self.load_sequence else (self.range[1] - self.range[0])
        sequence_length = (self.range[1] - self.range[0]) if self.load_sequence else 2
        sequence = []
        
        sample_idx = sample_info.idx_in_epoch

        if sample_idx >= len(self.folders):
            raise StopIteration()

        files = os.listdir(os.path.join(self.dir, self.folders[sample_idx]))
        files.sort()

        for k in range(sequence_length):
            file_dir = os.path.join(self.dir, self.folders[sample_idx], files[k * stride])
            with open(file_dir, 'rb') as f:
                grid = jnp.frombuffer(f.read(), dtype=jnp.float32)
                grid = grid.reshape(1, self.grid_size, self.grid_size, self.grid_size)
                delta = overdensity(grid)
                sequence.append(delta)

        return sequence
