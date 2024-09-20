def growth_factor_approx(a : float, Omega_M : float, Omega_L : float):

        return (5/2 * a * Omega_M) /\
            (Omega_M**(4 / 7) - Omega_L + (1 + Omega_M / 2) * (1 + Omega_L / 70 ))

def growth_factor_approx_deriv(Omega_M : float, Omega_L : float):

        return (5/2 * Omega_M) /\
            (Omega_M**(4 / 7) - Omega_L + (1 + Omega_M / 2) * (1 + Omega_L / 70 ))