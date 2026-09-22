"""Surrogate-based constrained reactor optimization."""
import time
import numpy as np
import pandas as pd
import joblib
from scipy.optimize import differential_evolution, minimize
from .config import BOUNDS, MODELS, RESULTS, INPUT_NAMES, OUTPUT_NAMES
from .reactor_model import solve_cstr

def load_surrogate():
    scalers = joblib.load(MODELS / "scalers.joblib")
    backend = scalers["backend"]
    if backend == "tensorflow":
        import tensorflow as tf
        model = tf.keras.models.load_model(MODELS / "surrogate_model.keras")
    else:
        model = joblib.load(MODELS / "surrogate_model.joblib")
    return model, scalers["x_scaler"], scalers["y_scaler"]

def predict(model, xs, ys, x):
    x_scaled = xs.transform(np.asarray(x).reshape(1, -1))
    
    try:
        y_scaled = model.predict(x_scaled, verbose=0)
    except TypeError:
        y_scaled = model.predict(x_scaled)
    return ys.inverse_transform(y_scaled)[0]

def objective(x, model, xs, ys):
    y = predict(model, xs, ys, x)
    T, CA, CB, CC = y
    # Penalize unsafe/physically implausible surrogate outputs.
    penalty = 0.0
    if T < 320 or T > 430:
        penalty += 1000.0 + 100.0 * abs(T - 375)
    if min(CA, CB, CC) < 0:
        penalty += 1000.0 * abs(min(CA, CB, CC))
    return -CB + penalty

def run_optimization(seed=42):
    RESULTS.mkdir(exist_ok=True)
    model, xs, ys = load_surrogate()
    bounds = [
        (BOUNDS["T_in"][0] + 5.0, BOUNDS["T_in"][1] - 5.0),
        (BOUNDS["C_A0"][0] + 0.1, BOUNDS["C_A0"][1] - 0.1),
        (BOUNDS["tau"][0] + 0.5, BOUNDS["tau"][1] - 0.5),
        (BOUNDS["V"][0] + 0.1, BOUNDS["V"][1] - 0.1),
    ]

    t0 = time.perf_counter()
    result = differential_evolution(
        lambda x: objective(x, model, xs, ys),
        bounds=bounds, seed=seed, popsize=12, maxiter=50, polish=True
    )
    surrogate_time = time.perf_counter() - t0

    x_opt = result.x
    y_sur = predict(model, xs, ys, x_opt)
    y_phys = solve_cstr(x_opt[0], x_opt[1], x_opt[2], x_opt[3], x_opt[3]/x_opt[2])

    direct = np.array([y_phys.T, y_phys.C_A, y_phys.C_B, y_phys.C_C])
    comparison = pd.DataFrame({
        "quantity": OUTPUT_NAMES,
        "surrogate": y_sur,
        "physics_model": direct,
        "absolute_error": np.abs(y_sur - direct)
    })
    comparison.to_csv(RESULTS / "optimization_validation.csv", index=False)

    opt_df = pd.DataFrame([x_opt], columns=INPUT_NAMES)
    for name, value in zip(OUTPUT_NAMES, y_sur):
        opt_df[f"pred_{name}"] = value
    opt_df["surrogate_optimization_time_s"] = surrogate_time
    opt_df.to_csv(RESULTS / "optimal_conditions.csv", index=False)

    print("\nOptimal operating conditions:")
    print(opt_df.T)
    print("\nValidation:")
    print(comparison)
    return x_opt, y_sur, direct

if __name__ == "__main__":
    run_optimization()
