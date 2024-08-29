from typing import NamedTuple

class CosmoParams(NamedTuple):
    box_size : float # Mpc / h
    n_grid : int
    redshift_start : int
    reshift_end : int
    