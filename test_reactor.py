import numpy as np
from src.reactor_model import solve_cstr, residuals

def test_reactor_solution_is_finite():
    x = solve_cstr(330.0, 1.2, 2.0, 0.5, 0.25)
    assert x.success
    assert np.all(np.isfinite([x.T, x.C_A, x.C_B, x.C_C]))
    assert x.C_A >= 0 and x.C_B >= 0 and x.C_C >= 0

def test_mass_balance_residual_small():
    x = solve_cstr(330.0, 1.2, 2.0, 0.5, 0.25)
    r = residuals([x.T, x.C_A, x.C_B, x.C_C], 330.0, 1.2, 2.0, 0.5, 0.25)
    assert np.linalg.norm(r) < 1e-4
