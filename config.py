"""Project-wide constants and reproducibility settings."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

SEED = 42

# Physical constants / reactor parameters
R = 8.314  # J/mol/K
RHO = 900.0  # kg/m3, representative liquid density
CP = 2500.0  # J/kg/K

# Consecutive reaction: A -> B -> C
K01 = 2.0e6       # 1/min
K02 = 5.0e5       # 1/min
E1 = 65000.0      # J/mol
E2 = 70000.0      # J/mol
DH1 = -8000.0    # J/mol
DH2 = -6000.0
UA = 100.0  # J/(L min K), effective jacket heat-removal coefficient
T_COOL = 300.0  # K    # J/mol

# Design/operating ranges
BOUNDS = {
    "T_in": (300.0, 360.0),       # K
    "C_A0": (0.8, 2.0),           # mol/L
    "tau": (0.5, 8.0),             # min
    "V": (0.05, 2.0),              # L
}
OUTPUT_NAMES = ["T", "C_A", "C_B", "C_C"]
INPUT_NAMES = list(BOUNDS.keys())

def set_seed(seed: int = SEED):
    np.random.seed(seed)
