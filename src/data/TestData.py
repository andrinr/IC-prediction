# from __future__ import annotations, type
import jax.numpy as jnp
import nvidia.dali.types as types

class TestData:
    def __init__(
            self, 
            batch_size : int, 
            grid_size : int):
        
        self.batch_size = batch_size
        self.grid_size = grid_size

    def __call__(self, sample_info : types.SampleInfo):

        if sample_info.idx_in_epoch > 10:
            raise StopIteration()
        
        sequence = []

        N = self.grid_size

        start = jnp.zeros((1, self.grid_size, self.grid_size, self.grid_size))

        end = jnp.zeros((1, self.grid_size, self.grid_size, self.grid_size))

        end = end.at[0, N // 4:3 * N // 4, N // 4:3 * N // 4, N // 4:3 * N // 4].set(1)

        sequence.append(start)
        sequence.append(end)
        
        return sequence
