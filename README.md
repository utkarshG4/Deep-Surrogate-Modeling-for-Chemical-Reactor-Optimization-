# Deep Surrogate Modeling for Chemical Reactor Optimization

## 📌 Project Overview
This repository introduces a physics-informed deep learning framework focused on **Deep Surrogate Modeling** to accelerate the simulation and real-time optimization of chemical reactors. Traditional chemical process optimization relies heavily on solving complex, coupled systems of non-linear differential equations (e.g., mass, energy, and momentum balances). While highly accurate, these mechanistic models (like rigorous CFD or Aspen Plus simulations) are computationally expensive and impractical for real-time control loops.

This project bridges that gap by training **Deep Neural Networks (DNNs)** as high-fidelity surrogate models. The trained deep surrogate acts as a "black-box" digital twin, predicting reactor outputs (conversion rates, temperature profiles, yield, selectivity) in milliseconds with near-mechanistic accuracy. This enables rapid multi-objective optimization, allowing operators to safely maximize yield while minimizing energy expenditure.

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Deep Learning Frameworks:** PyTorch / TensorFlow (Choose your active framework)
* **Optimization & Math:** SciPy (Optimization module), NumPy, Pandas
* **Visualization:** Matplotlib, Seaborn
* **Mechanistic Data Generation:** (Optional: Mention if you used Aspen Plus, MATLAB, or native Python ODE solvers to generate data)

## 📁 Repository Structure
* `src/` - Core source scripts containing neural network architectures, custom loss functions, and optimization algorithms.
* `notebooks/` - Step-by-step Jupyter Notebooks demonstrating data processing, model training, validation curves, and optimization runs.
* `data/` - Training, validation, and testing datasets generated from rigorous reactor simulation sweeps.
* `models/` - Saved weights (`.pt` or `.h5` files) and configurations for the trained deep surrogate networks.
* `results/` - Performance evaluations, parity plots comparing mechanistic vs. surrogate outputs, and optimization surface maps.
* `requirements.txt` - File detailing dependencies and library versions required to run the code.

## ⚙️ Core Methodologies & Engineering Workflow
1. **Data Generation & Sampling:** Generated high-dimensional datasets by sweeping key operational variables (e.g., space velocity, feed temperature, catalyst activity, inlet concentrations) using Latin Hypercube Sampling (LHS).
2. **Surrogate Architecture Selection:** Designed and trained Multi-Layer Perceptrons (MLPs) / Residual Networks (ResNets) to map non-linear input-output relations of the reactor.
3. **Loss Function Engineering:** Integrated physics-guided boundary constraints (e.g., preventing negative concentrations or ensuring mass balance approximations) into the neural network training loop.
4. **Surrogate Validation:** Evaluated accuracy using strict engineering metrics including Mean Absolute Percentage Error (MAPE), R² parity metrics, and maximum absolute error bounds.
5. **Real-time Optimization:** Coupled the fast-executing surrogate model with global/local optimization algorithms (e.g., Genetic Algorithms, Particle Swarm Optimization, or Sequential Least Squares Programming) to discover optimal reactor setpoints instantaneously.
