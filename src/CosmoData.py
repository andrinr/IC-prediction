import jax.numpy as jnp
from random import shuffle
import os

type Range = tuple[str, int]

class CosmosData(object):
    def __init__(
            self, 
            batch_size : int, 
            grid_size : int, 
            directory : str,
            range : Range,
            load_sequence : bool = False):

        self.dir = os.path.abspath(directory)
        self.batch_size = batch_size
        self.grid_size = grid_size
        self.range = range
        self.load_sequence = load_sequence
        self.folders = os.listdir(self.dir)
        shuffle(self.folders)

    def __iter__(self):
        self.i = 0  
        self.n = len(self.folders)
        return self

    def __next__(self):
        stride = 1 if self.load_sequence else (self.range[1] - self.range[0])
        sequence_length = (self.range[1] - self.range[0]) if self.load_sequence else 2
        batch = jnp.zeros(
            (self.batch_size, 
            sequence_length,
            self.grid_size, self.grid_size, self.grid_size))

        for j in range(self.batch_size):

            files = os.listdir(os.path.join(self.dir, self.folders[self.i]))
            files.sort()

            for k in range(sequence_length):
                
                file_dir = os.path.join(self.dir, self.folders[self.i], files[k * stride])
                with open(file_dir, 'rb') as f:
                    grid = jnp.frombuffer(f.read(), dtype=jnp.float32)
                    grid = grid.reshape(self.grid_size, self.grid_size, self.grid_size)
                    batch = batch.at[j, k].set(grid)
            
            self.i = (self.i + 1) % self.n

        print(jnp.shape(batch))
        return batch
