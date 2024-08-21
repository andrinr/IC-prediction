from jax import numpy as jnp
import numpy as np

header_type = jnp.dtype([('time', '>f8'),('N', '>i4'), ('Dims', '>i4'), ('Ngas', '>i4'), ('Ndark', '>i4'), ('Nstar', '>i4'), ('pad', '>i4')])
gas_type  = jnp.dtype([('mass','>f4'), ('x', '>f4'),('y', '>f4'),('z', '>f4'), ('vx', '>f4'),('vy', '>f4'),('vz', '>f4'),
	                ('rho','>f4'), ('temp','>f4'), ('hsmooth','>f4'), ('metals','>f4'), ('phi','>f4')])
dark_type = jnp.dtype([('mass','>f4'), ('x', '>f4'),('y', '>f4'),('z', '>f4'), ('vx', '>f4'),('vy', '>f4'),('vz', '>f4'),
	                ('eps','>f4'), ('phi','>f4')])
star_type  = jnp.dtype([('mass','>f4'), ('x', '>f4'),('y', '>f4'),('z', '>f4'), ('vx', '>f4'),('vy', '>f4'),('vz', '>f4'),
	                ('metals','>f4'), ('tform','>f4'), ('eps','>f4'), ('phi','>f4')])

def generate_tipsy(
        file_name : str,
        pos : np.ndarray,
        mass : np.ndarray,
        time : float = 0.0):
    """
    Generate a tipsy file.
    """

    N = pos.shape[0]

    header = np.zeros(1, dtype=header_type)
    header['time'] = time
    header['N'] = N
    header['Dims'] = 3
    header['Ngas'] = 0
    header['Ndark'] = N
    header['Nstar'] = 0

    # empty gas and star arrays
    gas = np.zeros(0, dtype=gas_type)
    dark = np.zeros(N, dtype=dark_type)
    star = np.zeros(0, dtype=star_type)

    # fill dark matter
    dark['mass'] = mass
    dark['x'] = pos[:, 0]
    dark['y'] = pos[:, 1]
    dark['z'] = pos[:, 2]

    with open(file_name, 'wb') as ofile:
        header.tofile(ofile)
        gas.tofile(ofile)
        star.tofile(ofile)