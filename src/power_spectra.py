from nbodykit.lab import *
import numpy as np
import matplotlib.pyplot as plt

cosmo = cosmology.Planck15

redshifts = [49, 10, 3, 0]

Plin = cosmology.LinearPower(cosmo, redshift=0, transfer='CLASS')
Pnl = cosmology.HalofitPower(cosmo, redshift=0)
Pzel = cosmology.ZeldovichPower(cosmo, redshift=0)

color_map = plt.get_cmap('inferno')
colors = color_map(np.linspace(0, 0.8, len(redshifts)))

for i, redshift in enumerate(redshifts):
    Plin.redshift = redshift
    Pnl.redshift = redshift
    Pzel.redshift = redshift

    k = np.logspace(-4, 1, 1000)

    Pk_lin = Plin(k)
    Pk_nl = Pnl(k)
    Pk_zel = Pzel(k)

    plt.plot(k, Pk_lin, label='z = %.2f Linear' % redshift, linestyle='--', color=colors[i])
    plt.plot(k, Pk_nl, label='z = %.2f Non-Linear' % redshift, linestyle='-.', color=colors[i])
    plt.plot(k, Pk_zel, label='z = %.2f Zeldovich' % redshift, linestyle='-', color=colors[i])

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$k$ [$h \ \mathrm{Mpc}^{-1}$]')
plt.ylabel(r'$P(k)$ [$h^{-3} \ \mathrm{Mpc}^3$]')
# place legend outside plot
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# make there is enough space for the legend

plt.tight_layout()

# figure size
plt.gcf().set_size_inches(8, 5)

# save the plot
plt.savefig('img/power_spectrum.png')
