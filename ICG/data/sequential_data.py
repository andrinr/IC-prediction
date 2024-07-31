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
    # https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/general/data_loading/external_input.html
    [sequence, steps] = fn.external_source(
        source=external_iterator,
        num_outputs=2,
        batch=False,
        dtype=types.FLOAT)

    reshape_fn = lambda x : fn.reshape(x, layout="FCDHW")
    resize_fn = lambda x : fn.resize(
        x,
        interp_type = types.INTERP_CUBIC,
        antialias=False,
        size=(grid_size, grid_size, grid_size))
    
    return resize_fn(reshape_fn(sequence)), steps

class VolumetricSequence:
    def __init__(
            self, 
            grid_size : int, 
            directory : str,
            start : int,
            steps : int,
            stride : int = None):

        self.dir = os.path.abspath(directory)
        self.grid_size = grid_size
        self.start = start
        self.steps = steps
        self.stride = steps if stride is None else stride
        self.folders = os.listdir(self.dir)
        shuffle(self.folders)

    def __call__(self, sample_info : types.SampleInfo):
        sequence_length = self.steps // self.stride + 1
        sequence = jnp.zeros(
            (sequence_length, 1, self.grid_size, self.grid_size, self.grid_size))
        
        sample_idx = sample_info.idx_in_epoch

        if sample_idx >= len(self.folders):
            raise StopIteration()

        files = os.listdir(os.path.join(self.dir, self.folders[sample_idx]))
        files.sort()

        for i in range(sequence_length):
            file_dir = os.path.join(
                self.dir, self.folders[sample_idx], files[self.start + i * self.stride])
            with open(file_dir, 'rb') as f:
                rho = jnp.frombuffer(f.read(), dtype=jnp.float32)
                rho = rho.reshape(1, self.grid_size, self.grid_size, self.grid_size)
                delta = overdensity(rho)
                sequence = sequence.at[i].set(delta)

        steps = jnp.linspace(self.start, self.stride * (sequence_length - 1), sequence_length)

        return list([sequence, steps])
