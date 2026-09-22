"""Generate final plots and direct-vs-surrogate speed comparison."""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .config import RESULTS, INPUT_NAMES, OUTPUT_NAMES, MODELS
from .reactor_model import solve_cstr
from .optimization import load_surrogate, predict

def make_plots():
    RESULTS.mkdir(exist_ok=True)
    pred_path = RESULTS / "test_predictions.csv"
    if not pred_path.exists():
        print("Run surrogate_model.py first.")
        return

    df = pd.read_csv(pred_path)
    for name in OUTPUT_NAMES:
        plt.figure(figsize=(6, 5))
        plt.scatter(df[f"{name}_actual"], df[f"{name}_pred"], s=12, alpha=0.55)
        lo = min(df[f"{name}_actual"].min(), df[f"{name}_pred"].min())
        hi = max(df[f"{name}_actual"].max(), df[f"{name}_pred"].max())
        plt.plot([lo, hi], [lo, hi], linestyle="--")
        plt.xlabel(f"Actual {name}")
        plt.ylabel(f"Predicted {name}")
        plt.title(f"Surrogate parity plot: {name}")
        plt.tight_layout()
        plt.savefig(RESULTS / f"parity_{name}.png", dpi=180)
        plt.close()

    hist = RESULTS / "training_history.csv"
    if hist.exists():
        h = pd.read_csv(hist)
        plt.figure(figsize=(7, 4))
        plt.plot(h["loss"], label="train")
        if "val_loss" in h:
            plt.plot(h["val_loss"], label="validation")
        plt.xlabel("Epoch")
        plt.ylabel("MSE loss")
        plt.title("Surrogate training history")
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULTS / "loss_curve.png", dpi=180)
        plt.close()

def speed_test(n=2000):
    model, xs, ys = load_surrogate()
    rng = np.random.default_rng(42)
    bounds = np.array([[300, 360], [0.8, 2.0], [0.5, 8.0], [0.05, 2.0]])
    X = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n, 4))

    t0 = time.perf_counter()
    _ = np.vstack([predict(model, xs, ys, x) for x in X[:min(500, n)]])
    surrogate_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = [solve_cstr(x[0], x[1], x[2], x[3], x[3]/x[2]) for x in X[:min(500, n)]]
    physics_time = time.perf_counter() - t0

    rows = pd.DataFrame([{
        "evaluations": min(500, n),
        "physics_time_s": physics_time,
        "surrogate_time_s": surrogate_time,
        "speedup": physics_time / surrogate_time if surrogate_time else np.nan
    }])
    rows.to_csv(RESULTS / "speed_comparison.csv", index=False)
    print(rows)
    return rows

if __name__ == "__main__":
    make_plots()
    speed_test()
