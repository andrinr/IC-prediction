from typing import NamedTuple

class Config(NamedTuple):
    grid_dir : str
    tipsy_dir : str
    model_dir : str
    output_tipsy_file : str
    model_type : str
    input_grid_size : int
    grid_size : int
    learning_rate : float
    file_index_stride : int | list[int]
    file_index_steps : int
    file_index_start : int
    redshifts : list[int]
    stepwise_epochs : int
    sequential_epochs : int
    unique_networks : bool
    flip : bool
    box_size : int 
    dt_PKDGRAV3 : float
    num_particles : int
    omega_L : float
    omega_M : float
