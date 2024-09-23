from typing import NamedTuple

class Config(NamedTuple):
    data_dir : str
    model_dir : str
    output_tipsy_file : str
    model_type : str
    input_grid_size : int
    grid_size : int
    learning_rate : float
    n_epochs : int
    stride : int
    steps : int
    sequential_mode : bool
    redshift_start : int
    redshift_end : int
    box_size : int 
    dt_PKDGRAV3 : float
    num_particles : int
    omega_L : float
    omega_M : float
