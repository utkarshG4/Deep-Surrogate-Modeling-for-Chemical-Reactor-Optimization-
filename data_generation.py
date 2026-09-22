"""Generate a reproducible synthetic reactor dataset."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import qmc
from .config import BOUNDS, DATA_RAW, DATA_PROCESSED, SEED, set_seed
from .reactor_model import simulate_row

def generate_dataset(n_samples=3000, seed=SEED):
    set_seed(seed)
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    names = list(BOUNDS)
    sampler = qmc.LatinHypercube(d=len(names), seed=seed)
    unit = sampler.random(n=n_samples)
    lower = np.array([BOUNDS[n][0] for n in names])
    upper = np.array([BOUNDS[n][1] for n in names])
    samples = qmc.scale(unit, lower, upper)

    rows = [simulate_row(*x) for x in samples]
    cols = names + ["T", "C_A", "C_B", "C_C", "success", "residual_norm"]
    df = pd.DataFrame(rows, columns=cols)
    df = df[df["success"]].copy()

    raw_path = DATA_RAW / "reactor_dataset_raw.csv"
    processed_path = DATA_PROCESSED / "reactor_dataset.csv"
    df.to_csv(raw_path, index=False)
    df.to_csv(processed_path, index=False)
    print(f"Generated {len(df)} valid simulations.")
    print(processed_path)
    return df

if __name__ == "__main__":
    generate_dataset()
