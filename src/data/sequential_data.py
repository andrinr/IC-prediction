# from __future__ import annotations, type
import jax.numpy as jnp
import os 
# NVIDIA Dali
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
# local
from cosmos import compute_overdensity

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
def volumetric_sequence_pipe(external_iterator, grid_size):
    
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
    
    normalize_fn = lambda x : fn.normalize(
        x,
        axes=[2, 3, 4],
        batch=True)
    
    return normalize_fn(resize_fn(reshape_fn(sequence))), steps, means

class VolumetricSequence:
    def __init__(
            self, 
            type : str,
            grid_size : int, 
            directory : str,
            start : int,
            steps : int,
            stride : int = None,
            flip : bool = True):
        
        """
        Loads all folder in a directory, 
        where each folder contains a sequence of 3D arrays, each stored as a binary.
        All sequences should have the same length. 
        Train, test and validation splits are not shuffled.
        """

        self.dir = os.path.abspath(directory)
        self.grid_size = grid_size
        self.start = start
        self.steps = steps
        self.stride = steps if stride is None else stride
        self.flip = flip
        self.folders = os.listdir(self.dir)

        b = int(TRAIN_SIZE * len(self.folders))
        c = b + int(VAL_SIZE * len(self.folders))

        if type == 'train':
            self.folders = self.folders[0 : b]
        
        if type == 'val':
            self.folders = self.folders[b : c]

        if type == 'test':
            self.folders = self.folders[c : -1]

    def __call__(self, sample_info : types.SampleInfo):
        sequence = jnp.zeros(
            (self.steps + 1, 1, self.grid_size, self.grid_size, self.grid_size))

        sample_idx = sample_info.idx_in_epoch

        if sample_idx >= len(self.folders):
            raise StopIteration()

        files = os.listdir(os.path.join(self.dir, self.folders[sample_idx]))
        files.sort()

        timeline = jnp.zeros(self.steps + 1)
        density_means = jnp.zeros(self.steps + 1)

        for i in range(self.steps + 1):
            time = self.start + i * self.stride

            timeline = timeline.at[i].set(time)
            file_dir = os.path.join(
                self.dir, self.folders[sample_idx], files[time])
        
            with open(file_dir, 'rb') as f:
                rho = jnp.frombuffer(f.read(), dtype=jnp.float32)
                rho = rho.reshape(1, self.grid_size, self.grid_size, self.grid_size)
                delta, mean = compute_overdensity(rho)
                density_means = density_means.at[i].set(mean)
                sequence = sequence.at[i].set(rho)
            
        if self.flip:
            sequence = jnp.flip(sequence, axis=0)
            timeline = jnp.flip(timeline, axis=0)
            density_means = jnp.flip(density_means, axis=0)

        return list([sequence, timeline, density_means])