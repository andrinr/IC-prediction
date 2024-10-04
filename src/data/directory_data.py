# from __future__ import annotations, type
import jax.numpy as jnp
import os 
# NVIDIA Dali
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.auto_aug.core import augmentation
# local
from cosmos import to_expansion, normalize

BATCH_SIZE = 2
TRAIN_SIZE = 0.8
TEST_SIZE = 0.05
VAL_SIZE = 0.05

@augmentation(mag_range=(0, 30), randomly_negate=True)
def rotate_aug(data, angle, fill_value=128, rotate_keep_size=True):
   return fn.rotate(data, angle=angle, fill_value=fill_value, keep_size=True)

@pipeline_def(
    batch_size=BATCH_SIZE,
    num_threads=2, 
    device_id=0,
    py_num_workers=16,
    py_start_method="spawn",
    enable_conditionals=True)
def directory_sequence_pipe(external_iterator, grid_size):
    
    [sequence, attributes] = fn.external_source(
        source=external_iterator,
        num_outputs=2,
        batch=False,
        dtype=types.FLOAT)

    # Frames, Channels, Depth, Height, Width
    reshape_fn = lambda x : fn.reshape(x, layout="FCDHW")
    resize_fn = lambda x : fn.resize(
        x,
        interp_type = types.INTERP_CUBIC,
        antialias=False,
        size=(grid_size, grid_size, grid_size))
    
    # shapes = fn.peek_image_shape(sequence)
    # sequence = rand_augment.rand_augment(sequence, shape=shapes, n=3, m=17)
    
    return resize_fn(reshape_fn(sequence)), attributes

class DirectorySequence:
    def __init__(
            self, 
            type : str,
            grid_size : int, 
            grid_directory : str,
            normalizing_function : str,
            start : int,
            steps : int,
            stride : int | list[int],
            flip : bool = True):
            
        """
        Loads all folder in a directory, 
        where each folder contains a sequence of 3D arrays, each stored as a binary.
        All sequences should have the same length. 
        Train, test and validation splits are not shuffled.
        """

        self.grid_dir = os.path.abspath(grid_directory)
        self.grid_size = grid_size
        self.start = start
        self.steps = steps
        self.stride = stride
        self.normalizing_function = normalizing_function
        if isinstance(self.stride, list): 
            self.stride = stride.copy()
            self.stride.append(0)
        self.flip = flip
        self.grid_folders = os.listdir(self.grid_dir)
        # self.tipsy_folders = os.listdir(self.tipsy_dir)

        self.grid_folders.sort()
        # self.tipsy_folders.sort()

        b = int(TRAIN_SIZE * len(self.grid_folders))
        c = b + int(VAL_SIZE * len(self.grid_folders))

        if type == 'train':
            self.grid_folders = self.grid_folders[0 : b]
            print(len(self.grid_folders))
            # self.tipsy_folders = self.tipsy_folders[0 : b]

        elif type == 'val':
            self.grid_folders = self.grid_folders[b : c]
            # self.tipsy_folders = self.tipsy_folders[0 : b]

        elif type == 'test':
            self.grid_folders = self.grid_folders[c : -1]
            # self.tipsy_folders = self.tipsy_folders[0 : b]
            
        else :
            raise NotImplementedError("type {type} not found")

    def __call__(self, sample_info : types.SampleInfo):
        sequence = jnp.zeros(
            (self.steps + 1, 1, self.grid_size, self.grid_size, self.grid_size))

        sample_idx = sample_info.idx_in_epoch

        if sample_idx >= len(self.grid_folders):
            raise StopIteration()

        grid_files = os.listdir(os.path.join(self.grid_dir, self.grid_folders[sample_idx]))
        grid_files.sort()

        attributes = []

        time = self.start
        for i in range(self.steps + 1):
            grid_file = os.path.join(
                self.grid_dir, self.grid_folders[sample_idx], grid_files[time])

            with open(grid_file, 'rb') as f:
                rho = jnp.frombuffer(f.read(), dtype=jnp.float32)
                rho = rho.reshape(1, self.grid_size, self.grid_size, self.grid_size)
                rho *= 2.777 * 10**11
                rho += 0.0001

                a = to_expansion(time/len(grid_files))

                normalized, attributes_ = normalize(rho, a, self.normalizing_function)

                attributes.append(attributes_)

                sequence = sequence.at[i].set(normalized)

            if isinstance(self.stride, list): 
                time += self.stride[i]
            else:
                time =+ self.stride

        if self.flip:
            sequence = jnp.flip(sequence, axis=0)
            attributes = jnp.flip(jnp.array(attributes), axis=0)

        return list([sequence, attributes])