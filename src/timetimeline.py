
from cosmology import SimpleCosmology

h               = 0.704000000000000
dOmega0         = 0.270877000000000
dLambda         = 1 - dOmega0

csm = SimpleCosmology(omega_matter=dOmega0,omega_lambda=dLambda,h=h)

z = 0 # redshift
a = 1.0 / (1.0 + z) # expansion factor

a = csm.time_to_hubble(0.1)
# (1.0 + z) * a = 1.0
# a + z * a = 1.0
# z = (1.0 - a) / a
z = 1.0 / a - 1.0
print(a, z)

# Then delta-T (the step size) to the given redshift ‘z’ is just ’t’.

# dpotter@studio pkdgrav3 % python t49.py
# 0.02 0.0012517188943844037