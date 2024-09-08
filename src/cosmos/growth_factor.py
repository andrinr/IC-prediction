def compute_growth_factor_deriv(
    Omega_M : float,
    Omega_L : float) -> float:
    """
    Compute derivative of the growth factor.
    """
    return  5 / 2 * Omega_M / (Omega_M**(4/7) - Omega_L + (1 + Omega_M / 2) * (1 + Omega_L / 70))

def compute_growth_factor(
    a : float,
    Omega_M : float,
    Omega_L : float) -> float:
    """
    Compute the growth factor at a given scale factor a.
    """

    D_plus = compute_growth_factor_deriv(Omega_M, Omega_L) * a

    return D_plus