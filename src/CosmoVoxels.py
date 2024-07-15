import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import os

import jax.numpy as jnp

class CosmoVoxels:
    def __init__(
        self, 
        directory : str,
        start_with : str,
        num_items : int, 
        transform=None, 
        target_transform=None):

        self.directory = directory
        self.start_with = start_with
        self.num_items = num_items
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.num_items)

    def __getitem__(self, idx : int):
        
        path = os.path.join(self.directory, str((idx+1)).zfill(3))

        file_names = os.listdir(path)
        file_names.sort()
        file_names = [s for s in file_names if s.startswith(self.start_with)]

        tensor = jnp.zeros()
        for file in file_names:
