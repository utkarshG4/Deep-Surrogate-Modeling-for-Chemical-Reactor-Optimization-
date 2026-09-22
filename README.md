# Deep Surrogate Modeling for Exothermic CSTR Optimization

## 📌 Project Overview
This repository delivers an integrated **Chemical Engineering and Deep Learning framework** designed for the high-fidelity optimization of a nonlinear, exothermic Continuous Stirred-Tank Reactor (CSTR). The system models the consecutive reaction kinetics of **A → B → C**, where **B** represents the highly desired target product. 

By replacing computationally demanding, iterative first-principles numerical models with a multi-output **Deep Neural Network (DNN) surrogate**, this workflow achieves significant computational acceleration. The resulting digital twin framework enables instantaneous, real-time multi-variable process design workflows and optimization loops without sacrificing the underlying physical rigor of the system.

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Deep Learning Framework:** TensorFlow / Keras
* **Data Processing & Analytics:** Scikit-learn, Pandas, NumPy
* **Numerical Modeling & Optimization:** SciPy (Differential Evolution)
* **Visualization:** Matplotlib, Seaborn

## 📁 Repository Structure
* `src/` - Core Python modules containing the physics-based ODE reactor simulator, custom DNN architectures, and optimization logic.
* `notebooks/` - Jupyter Notebooks mapping out exploratory data sweeps, surrogate network training phases, and optimization convergence.
* `data/` - Training, validation, and benchmarking datasets generated via Latin Hypercube Sampling.
* `models/` - Saved weights, model architectures, and serialized configurations for the trained TensorFlow/Keras neural networks.
* `results/` - Validation metrics, parity plots, computational acceleration benchmarks, and verified reactor setpoints.
* `requirements.txt` - Configuration file detailing library dependencies for easy environment replication.

## ⚙️ Core Methodologies & Engineering Workflow
1. **First-Principles Modeling:** Developed a rigorous, physics-based CSTR model using material balances, energy balances, and temperature-dependent **Arrhenius reaction kinetics**.
2. **Advanced Data Generation:** Employed **Latin Hypercube Sampling (LHS)** to execute thousands of physical reactor simulations, uniformly mapping out the operational boundary spaces.
3. **Deep Surrogate Architecture:** Built and trained a multi-output **Deep Neural Network (DNN)** to map operating and design inputs directly to critical steady-state outputs, including **reactor temperature and species concentrations**.
4. **Rigorous Validation:** Evaluated surrogate predictive precision and generalization limits against unseen validation data using **Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R² scores**, and detailed visual **parity plots**.
5. **Metaheuristic Optimization:** Integrated the rapid-executing DNN surrogate with a constrained **Differential Evolution algorithm** to discover the precise operating conditions that maximize the concentration of desired product **B**.
6. **Physics Verification & Benchmarking:** Cross-verified the optimal machine learning setpoints by running them back through the original mechanistic model, followed by a formal computational benchmark validating the order-of-magnitude processing speedup.
