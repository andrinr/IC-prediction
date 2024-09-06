def compute_growth_factor(
    a : float,
    Omega_M : float,
    Omega_L : float) -> float:
    """
    Compute the growth factor at a given scale factor a.
    """

    D_plus = 5 / 2 * a * Omega_M / (Omega_M**(4/7) - Omega_L + (1 + Omega_M / 2) * (1 + Omega_L / 70))

    return D_plus