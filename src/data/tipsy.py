import numpy as np
import struct

header_type = np.dtype([('time', '>f8'),('N', '>i4'), ('Dims', '>i4'), ('Ngas', '>i4'), ('Ndark', '>i4'), ('Nstar', '>i4'), ('pad', '>i4')])
# gas_type  = np.dtype([('mass','>f4'), ('r', '>f4', [3,]), ('v', '>f4',[3,]),
#                       ('rho','>f4'), ('temp','>f4'), ('hsmooth','>f4'), ('metals','>f4'), ('phi','>f4')])
dark_type = np.dtype([('mass','>f4'), ('r', '>f4', [3,]), ('v', '>f4',[3,]),
                      ('eps','>f4'), ('phi','>f4')])
# star_type  = np.dtype([('mass','>f4'), ('r', '>f4', [3,]), ('v', '>f4',[3,]),
#                        ('metals','>f4'), ('tform','>f4'), ('eps','>f4'), ('phi','>f4')])

# od --endian=big -j 0 _N 8 -t f8 / d4
def generate_tipsy(
        file_name : str,
        pos : np.ndarray,
        vel : np.ndarray,
        mass : np.ndarray,
        time : float):
    """
    Generate a tipsy file.
    """

    pos = pos.astype(dtype=np.float32)
    vel = vel.astype(dtype=np.float32)
    mass = mass.astype(dtype=np.float32)

    N = pos.shape[1]

    header = np.zeros(1, dtype=header_type)
    header['time'] = time
    header['N'] = N
    header['Dims'] = 3
    header['Ngas'] = 0
    header['Ndark'] = N
    header['Nstar'] = 0

    # empty gas and star arrays
    # gas = np.zeros(0, dtype=gas_type)
    dark = np.zeros(N, dtype=dark_type)
    # star = np.zeros(0, dtype=star_type)

    # fill dark matter
    dark['r'] = np.vstack([pos[0, :].ravel(), pos[1, :].ravel(), pos[2, :].ravel()]).T
    dark['v'] = np.vstack([vel[0, :].ravel(), vel[1, :].ravel(), vel[2, :].ravel()]).T
    dark['mass'] = mass

    dark['eps'] = np.zeros(N, dtype=np.float32) * 1 / (N * 50)
    # phi stays zero

    with open(file_name, 'wb') as ofile:
        header.tofile(ofile)
        # gas.tofile(ofile)
        dark.tofile(ofile)
        # star.tofile(ofile)
    