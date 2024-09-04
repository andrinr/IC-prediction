import numpy as np
from .tipsy import generate_tipsy

def test_tipsy():

    pos = np.ones((100, 3))
    mass = np.ones((100, ))

    generate_tipsy(
        'test.tipsy', pos=pos, mass=mass, time=0)
    
    assert True