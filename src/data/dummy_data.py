# from __future__ import annotations, type
import jax.numpy as jnp
import nvidia.dali.types as types
import random

class DummyData:
    def __init__(
            self, 
            batch_size : int, 
            grid_size : int):
        
        self.batch_size = batch_size
        self.grid_size = grid_size

    def __call__(self, sample_info : types.SampleInfo):

        if sample_info.idx_in_epoch > self.batch_size * 10:
            raise StopIteration()
        
        sequence = []

        N = self.grid_size

        square_size = N // 6
        square_center = random.randint(N // 4, 3 * N // 4)

        a = jnp.zeros((1, self.grid_size, self.grid_size, self.grid_size))
        a = a.at[0, 
                :, 
                square_center - square_size:square_center + square_size, 
                square_center - square_size:square_center + square_size].set(1)
        
        # square_center += 0

        # b = jnp.zeros((1, self.grid_size, self.grid_size, self.grid_size))
        # b = b.at[0, 
        #         :, 
        #         square_center - square_size:square_center + square_size, 
        #         square_center - square_size:square_center + square_size].set(1)

        sequence.append(a)
        sequence.append(a)
        
        return sequence
