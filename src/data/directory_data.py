# from __future__ import annotations, type
import jax.numpy as jnp
import os 
# NVIDIA Dali
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.auto_aug.core import augmentation
from nvidia.dali.auto_aug import rand_augment
# local
from cosmos import compute_overdensity
from .tipsy import read_tipsy
from field import cic_ma
from .normalize import normalize

BATCH_SIZE = 4
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
    
    [sequence, steps, attributes] = fn.external_source(
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
    
    # shapes = fn.peek_image_shape(sequence)
    # sequence = rand_augment.rand_augment(sequence, shape=shapes, n=3, m=17)
    
    return resize_fn(reshape_fn(sequence)), steps, attributes

class DirectorySequence:
    def __init__(
            self, 
            type : str,
            grid_size : int, 
            grid_directory : str,
            tipsy_directory : str,
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

        self.grid_dir = os.path.abspath(grid_directory)
        self.tipsy_dir = os.path.abspath(tipsy_directory)
        self.grid_size = grid_size
        self.start = start
        self.steps = steps
        self.stride = steps if stride is None else stride
        self.flip = flip
        self.grid_folders = os.listdir(self.grid_dir)
        # self.tipsy_folders = os.listdir(self.tipsy_dir)

        self.grid_folders.sort()
        # self.tipsy_folders.sort()

        b = int(TRAIN_SIZE * len(self.grid_folders))
        c = b + int(VAL_SIZE * len(self.grid_folders))

        if type == 'train':
            self.grid_folders = self.grid_folders[0 : b]
            # self.tipsy_folders = self.tipsy_folders[0 : b]

        if type == 'val':
            self.grid_folders = self.grid_folders[b : c]
            # self.tipsy_folders = self.tipsy_folders[0 : b]

        if type == 'test':
            self.grid_folders = self.grid_folders[c : -1]
            # self.tipsy_folders = self.tipsy_folders[0 : b]

    def __call__(self, sample_info : types.SampleInfo):
        sequence = jnp.zeros(
            (self.steps + 1, 1, self.grid_size, self.grid_size, self.grid_size))

        sample_idx = sample_info.idx_in_epoch

        if sample_idx >= len(self.grid_folders):
            raise StopIteration()

        grid_files = os.listdir(os.path.join(self.grid_dir, self.grid_folders[sample_idx]))
        grid_files.sort()

        # tipsy_files = os.listdir(os.path.join(self.tipsy_dir, self.tipsy_folders[sample_idx]))
        # tipsy_files.sort()

        timeline = jnp.zeros(self.steps + 1)
        attributes = jnp.zeros((self.steps + 1, 2)) # min, max

        for i in range(self.steps + 1):
            time = self.start + i * self.stride

            timeline = timeline.at[i].set(time)
            grid_file = os.path.join(
                self.grid_dir, self.grid_folders[sample_idx], grid_files[time])
            
            # tipsy_file = os.path.join(
            #     self.tipsy_dir, self.grid_folders[sample_idx], grid_files[time])
            
            # header, dark = read_tipsy(tipsy_file)
            # pos = jnp.ndarray([dark["x"], dark["y"], dark["z"]])
            # pos = pos + 1.0
            # vx = dark["vx"]
            # vy = dark["vx"]
            # vz = dark["vz"]

            # field_vx = cic_ma(pos, vx, self.grid_size)
            # field_vy = cic_ma(pos, vx, self.grid_size)
            # field_vz = cic_ma(pos, vz, self.grid_size)

            # timeline = timeline.at[i].set(header["time"])
            # timeline = timeline.at[i].set(header["time"])

            with open(grid_file, 'rb') as f:
                rho = jnp.frombuffer(f.read(), dtype=jnp.float32)
                rho = rho.reshape(1, self.grid_size, self.grid_size, self.grid_size)
                rho *= 2.777 * 10**11
                rho += 0.0001

                rho, min, max = normalize(rho)

                attributes = attributes.at[i, 0].set(min)
                attributes = attributes.at[i, 1].set(max)

                sequence = sequence.at[i].set(rho)
            
        if self.flip:
            sequence = jnp.flip(sequence, axis=0)
            timeline = jnp.flip(timeline, axis=0)
            attributes = jnp.flip(attributes, axis=0)

        return list([sequence, timeline, attributes])