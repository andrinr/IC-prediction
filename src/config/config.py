from typing import NamedTuple

class Config(NamedTuple):
    train_data_dir : str
    test_data_dir : str
    input_grid_size : int
    grid_size : int
    learning_rate : float
    n_epochs : int
    stride : int
    model_params_file : str