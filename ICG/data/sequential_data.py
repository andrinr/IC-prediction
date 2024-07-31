# from __future__ import annotations, type
import jax.numpy as jnp
import jax
from random import shuffle
import os
# NVIDIA Dali
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types

type Range = tuple[str, int]

BATCH_SIZE = 8

def overdensity(density):
    mean = density.mean()
    return (density - mean) / mean

@pipeline_def(
    batch_size=BATCH_SIZE,
    num_threads=2, 
    device_id=0,
    py_num_workers=16,
    py_start_method="spawn")
def volumetric_sequence_pipe(external_iterator, grid_size):

    [start, end] = fn.external_source(
        source=external_iterator,
        num_outputs=2,
        batch=False,
        dtype=types.FLOAT)

    reshape_fn = lambda x : fn.reshape(x, layout="CDHW")
    resize_fn = lambda x : fn.resize(
        x,
        interp_type = types.INTERP_CUBIC,
        antialias=False,
        size=(grid_size, grid_size, grid_size))

    start = resize_fn(reshape_fn(start))
    end = resize_fn(reshape_fn(end))
    
    return start, end

class VolumetricSequence:
    def __init__(
            self, 
            grid_size : int, 
            directory : str,
            start : int | str, # start index of the sequence, can be 'random'
            end : int,
            steps : int,
            stride : int = None):

        self.dir = os.path.abspath(directory)
        self.grid_size = grid_size
        self.start = start
        if start == 'random':
            self.start = jax.random.randint(0, end - steps)
        self.steps = steps
        self.stride = steps if stride is None else stride
        self.folders = os.listdir(self.dir)
        shuffle(self.folders)

    def __call__(self, sample_info : types.SampleInfo):
        sequence_length = self.steps // self.stride
        sequence = []
        
        sample_idx = sample_info.idx_in_epoch

        if sample_idx >= len(self.folders):
            raise StopIteration()

        files = os.listdir(os.path.join(self.dir, self.folders[sample_idx]))
        files.sort()

        for k in range(sequence_length):
            file_dir = os.path.join(
                self.dir, self.folders[sample_idx], files[self.start + k * self.stride])
            with open(file_dir, 'rb') as f:
                grid = jnp.frombuffer(f.read(), dtype=jnp.float32)
                grid = grid.reshape(1, self.grid_size, self.grid_size, self.grid_size)
                delta = overdensity(grid)
                sequence.append(delta)

        return sequence
