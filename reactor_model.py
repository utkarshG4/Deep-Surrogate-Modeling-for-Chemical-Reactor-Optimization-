"""Physics-based nonlinear steady-state CSTR model."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq
from .config import R, RHO, CP, K01, K02, E1, E2, DH1, DH2, UA, T_COOL

@dataclass
class ReactorResult:
    T: float
    C_A: float
    C_B: float
    C_C: float
    success: bool
    residual_norm: float

def rate_constants(T):
    k1 = K01 * np.exp(-E1 / (R * T))
    k2 = K02 * np.exp(-E2 / (R * T))
    return k1, k2

def concentrations_at_T(T, C_A0, tau):
    k1, k2 = rate_constants(T)
    CA = C_A0 / (1.0 + k1 * tau)
    CB = tau * k1 * CA / (1.0 + k2 * tau)
    CC = tau * k2 * CB
    return CA, CB, CC, k1, k2

def energy_residual(T, T_in, C_A0, tau):
    CA, CB, CC, k1, k2 = concentrations_at_T(T, C_A0, tau)
    r1 = k1 * CA
    r2 = k2 * CB
    heat_release = (-DH1) * r1 * 1000.0 + (-DH2) * r2 * 1000.0
    energy = (RHO * CP / 1000.0) * (T_in - T) / tau + heat_release - UA * (T - T_COOL)
    return energy / 1.0e6

def residuals(z, T_in, C_A0, tau, V, Q):
    T, CA, CB, CC = z
    r1, r2 = rate_constants(T)[0] * CA, rate_constants(T)[1] * CB
    return np.array([
        (C_A0 - CA) / tau - r1,
        -CB / tau + r1 - r2,
        -CC / tau + r2,
        energy_residual(T, T_in, C_A0, tau),
    ])

def solve_cstr(T_in, C_A0, tau, V, Q, x0=None):
    """Solve the steady-state CSTR using a robust 1-D temperature root.

    Concentration balances are solved analytically for a trial temperature,
    then the nonlinear energy balance is solved by bracketing.
    """
    # Search the physical temperature range for a sign-changing root.
    grid = np.linspace(250.0, 500.0, 501)
    vals = np.array([energy_residual(T, T_in, C_A0, tau) for T in grid])
    candidates = []
    for i in range(len(grid) - 1):
        if np.isfinite(vals[i]) and vals[i] == 0:
            candidates.append(grid[i])
        elif np.isfinite(vals[i]) and np.isfinite(vals[i+1]) and vals[i] * vals[i+1] < 0:
            candidates.append(brentq(lambda T: energy_residual(T, T_in, C_A0, tau), grid[i], grid[i+1]))
    if not candidates:
        # Use the minimum-energy-residual temperature if no bracketed root exists.
        T = float(grid[np.argmin(np.abs(vals))])
        success = False
    else:
        # Choose the lowest-temperature physical steady state; this is a
        # deterministic convention when the exothermic model has multiplicity.
        T = float(min(candidates))
        success = True
    CA, CB, CC, _, _ = concentrations_at_T(T, C_A0, tau)
    r = residuals([T, CA, CB, CC], T_in, C_A0, tau, V, Q)
    return ReactorResult(T, CA, CB, CC, success, float(np.linalg.norm(r)))

def simulate_row(T_in, C_A0, tau, V):
    Q = V / tau
    result = solve_cstr(T_in, C_A0, tau, V, Q)
    return [T_in, C_A0, tau, V, result.T, result.C_A, result.C_B, result.C_C,
            result.success, result.residual_norm]
