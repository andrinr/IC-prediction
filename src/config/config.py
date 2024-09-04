from typing import NamedTuple

class Config(NamedTuple):
    train_data_dir : str
    test_data_dir : str
    model_params_file : str
    output_tipsy_file : str
    input_grid_size : int
    grid_size : int
    learning_rate : float
    n_epochs : int
    stride : int
    redshift_start : int
    redshift_end : int
    box_size : int 
    dt_PKDGRAV3 : float
    num_particles : int
