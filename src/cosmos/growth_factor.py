import jax.numpy as jnp

"""
As of now this are all pretty much magic functions which I don't understand.
"""

def compute_Omega_K(
        Omega_M : float,
        Omega_L : float) -> float:
    """
    Compute the curvature density parameter.
    """
    return 1 - Omega_M - Omega_L

def compute_eta(a : float, Omega_M : float, Omega_L : float) -> float:
    """
    Compute the conformal time eta at a given scale factor a.
    """
    Omega_K = compute_Omega_K(Omega_M, Omega_L)
    return jnp.sqrt(Omega_M / a + Omega_L * a**2 + Omega_K)

def growth_integrand(a : float, Omega_M : float, Omega_L : float) -> float:

    eta = compute_eta(a, Omega_M, Omega_L)

    return 2.5 / (eta ** 3)

def compute_growth_factor(
    a : float,
    Omega_M : float,
    Omega_L : float) -> float:
    """
    Compute the growth factor at a given scale factor a.
    """

    eta = compute_eta(a, Omega_M, Omega_L)
    
    integrate_from = 1.0e-10
    integrate_to = a

    x = jnp.linspace(integrate_from, integrate_to, 1000)
    y = growth_integrand(x, Omega_M, Omega_L)

    integral = jnp.trapz(y, x)

    return eta / a * integral

def compute_v_factor(
        a : float,
        Omega_M : float,
        Omega_L : float) -> float:
    
    Omega_K = compute_Omega_K(Omega_M, Omega_L)

    Dplus = compute_growth_factor(a, Omega_M, Omega_L)

    eta = compute_eta(a, Omega_M, Omega_L)

    f_omega  = (2.5 / Dplus - 1.5 * Omega_M / a - Omega_K) / eta / eta

    d_log_a_dt = a * eta

    return f_omega / d_log_a_dt / a * 100.0