# from __future__ import annotations, type
import jax.numpy as jnp
import random
# NVIDIA Dali
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types

BATCH_SIZE = 4
TRAIN_SIZE = 0.8
TEST_SIZE = 0.05
VAL_SIZE = 0.05

@pipeline_def(
    batch_size=BATCH_SIZE,
    num_threads=2, 
    device_id=0,
    py_num_workers=16,
    py_start_method="spawn")
def cube_sequence_pipe(external_iterator, grid_size):
    
    [sequence, steps, means] = fn.external_source(
        source=external_iterator,
        num_outputs=3,
        batch=False,
        dtype=types.FLOAT)

    # Frames, Channels, Depth, Height, Width
    reshape_fn = lambda x : fn.reshape(x, layout="FCDHW")
    resize_fn = lambda x : fn.resize(
        x,
        interp_type = types.INTERP_CUBIC,
        antialias=False,
        size=(grid_size, grid_size, grid_size))
    
    return resize_fn(reshape_fn(sequence)), steps, means

class CubeData:
    def __init__(
            self, 
            batch_size : int, 
            steps : int,
            grid_size : int):
        
        self.batch_size = batch_size
        self.grid_size = grid_size
        self.steps = steps

    def __call__(self, sample_info : types.SampleInfo):

        if sample_info.idx_in_epoch > self.batch_size * 10:
            raise StopIteration()
        
        sequence = jnp.zeros(
            (self.steps + 1, 1, self.grid_size,  self.grid_size,  self.grid_size))

        N = self.grid_size

        square_size = N // 6
        square_center = random.randint(N // 4, 3 * N // 4)

        for i in range(self.steps + 1):

            sequence = sequence.at[i, 
                0,
                :,
                square_center - square_size:square_center + square_size,
                square_center - square_size:square_center + square_size].set(1)

            
            square_center += 10
            square_size -= 4

        
        return list([sequence, jnp.linspace(0, self.steps, self.steps + 1), jnp.zeros(self.steps)])
