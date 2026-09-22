# Deep Surrogate Modeling for Chemical Reactor Optimization

An intermediate chemical-engineering + deep-learning project that builds a physics-based CSTR simulator, generates a design-of-experiments dataset, trains a deep neural-network surrogate, and uses the surrogate for constrained reactor optimization.

## Project objective

For the consecutive exothermic reaction

\[
A \rightarrow B \rightarrow C
\]

the desired product is **B**. The project learns a surrogate mapping from reactor operating conditions to steady-state reactor outputs and then optimizes the operating conditions for maximum B concentration / yield.

### Workflow

```text
Physics-based CSTR model
        |
        v
Latin Hypercube sampling
        |
        v
Synthetic simulation dataset
        |
        v
Preprocessing + EDA
        |
        v
Deep neural-network surrogate
        |
        v
Prediction/error analysis
        |
        v
Constrained surrogate optimization
        |
        v
Validation using original reactor model
        |
        v
Computational speed comparison
```

## Engineering model

The steady-state CSTR balances are:

\[
0 = F(C_{A0}-C_A)-Vk_1C_A
\]

\[
0 = -FC_B+V(k_1C_A-k_2C_B)
\]

\[
0 = -FC_C+Vk_2C_B
\]

with \(F=Q\) and \(\tau=V/F\).

For an exothermic reaction, the energy balance is:

\[
0 = \rho C_p F(T_{in}-T)
+V(-\Delta H_1)k_1C_A
+V(-\Delta H_2)k_2C_B
\]

The Arrhenius rate constants are:

\[
k_i=k_{0,i}\exp\left(-\frac{E_i}{RT}\right)
\]

The model uses physically plausible bounds and a bounded nonlinear solver to obtain the steady state.

## Machine-learning model

The surrogate is a multi-output feed-forward neural network:

```text
4 inputs
   |
Dense(64) + ReLU
   |
Dense(128) + ReLU
   |
Dense(64) + ReLU
   |
Dense(32) + ReLU
   |
4 outputs
```

Inputs:

- inlet temperature
- inlet concentration
- residence time
- reactor volume

Flow rate is derived physically from `Q = V/tau`, avoiding redundant/inconsistent inputs.

Outputs:

- reactor temperature
- A concentration
- B concentration
- C concentration

The implementation uses TensorFlow/Keras when available. The code also contains a scikit-learn MLP fallback so the project can still run in lightweight environments.

## Optimization

The surrogate is used to maximize desired-product concentration while enforcing:

- temperature safety bounds
- positive species concentrations
- residence-time limits
- feed/flow constraints

The optimized point is always checked against the original physics model.

## Repository structure

```text
deep-surrogate-reactor-optimization/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── notebooks/
│   └── 00_complete_project.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── reactor_model.py
│   ├── data_generation.py
│   ├── preprocessing.py
│   ├── surrogate_model.py
│   ├── optimization.py
│   └── validation.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── results/
└── tests/
    └── test_reactor.py
```

## Run locally

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
python -m src.data_generation
python -m src.surrogate_model
python -m src.optimization
python -m src.validation
```

Or open `notebooks/00_complete_project.ipynb` in Jupyter/Google Colab and run the sections in order.

## Expected outputs

The pipeline creates:

- `data/processed/reactor_dataset.csv`
- `models/surrogate_model.keras` when TensorFlow is installed
- `models/surrogate_model.joblib` for the scikit-learn fallback
- parity plots
- loss curves
- residual plots
- optimization results
- direct-model vs surrogate timing comparison

## Why this is useful

Repeated optimization of a nonlinear reactor can require thousands of model evaluations. A trained surrogate can approximate the expensive simulation and provide fast predictions during the optimization search. The final candidate is then re-simulated with the original physics model to avoid treating the surrogate as ground truth.

## Reproducibility

Random seeds are fixed in `src/config.py`. The dataset is generated from the physics model rather than being downloaded from an external source.

## Resume-ready project statement

**Deep Surrogate Modeling for Chemical Reactor Optimization** — Python, TensorFlow/Keras, SciPy, Scikit-learn

- Developed a physics-based nonlinear CSTR model for consecutive exothermic reactions and generated a synthetic operating dataset using Latin Hypercube sampling.
- Trained a multi-output deep neural-network surrogate to predict reactor temperature and species concentrations from operating conditions.
- Integrated the surrogate with constrained numerical optimization to maximize desired-product concentration and validated the optimum against the original reactor model.
- Quantified surrogate accuracy, residual error, and computational speed-up for repeated reactor evaluations.
