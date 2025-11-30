# ⚡ Superconducting Diode Effect Simulator

> **A Python-based framework for simulating non-reciprocal transport in superconducting thin films using the Time-Dependent Ginzburg-Landau (TDGL) equations.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TDGL](https://img.shields.io/badge/Solver-py--tdgl-green)](https://github.com/loganbvh/py-tdgl)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange?logo=jupyter)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)]()

## 📖 Overview

This project provides a comprehensive computational environment to explore the **Superconducting Diode Effect (SDE)**—where a superconductor exhibits different critical currents depending on the direction of current flow.

Built on top of the powerful `py-tdgl` library, this repository allows researchers and enthusiasts to model asymmetric superconducting geometries (such as weak links or notched bridges) and analyze their transport properties under varying magnetic fields and currents.

## 🚀 Key Features

* **Asymmetric Device Generation:** Easily create custom geometries (e.g., T-shapes, notched bridges) to induce symmetry breaking required for the diode effect.
* **Automated Simulations:**
    * **IV Characteristics:** Trace Voltage vs. Current curves to identify critical currents ($I_{c+}$ and $I_{c-}$).
    * **Field Dependence:** Analyze how the diode efficiency evolves under different external magnetic fields.
    * **Magnetization:** Study the magnetic response of the superconducting film.
* **Dynamic Visualization:** Generate animations of the Order Parameter $|\psi|^2$ to visualize vortex dynamics and phase slips in real-time.
* **HPC Ready:** Configurations available for running simulations on CPUs or accelerating with CUDA (GPU) if supported.

## 📂 Repository Structure

* **`proyecto_electro2.ipynb`** The main entry point. A Jupyter Notebook that orchestrates the simulation workflow. It includes:
    * Parameter setup ($ \xi $, $ \lambda $, $\Gamma$).
    * Execution of simulation loops (Varying Currents, Varying Fields).
    * Data plotting and visualization.

* **`default_functions.py`** A modular helper script containing the core logic:
    * `create_device()`: Defines the mesh and geometry of the superconducting bridge.
    * `default_solution()`: Wraps the solver execution logic.
    * `varying_increments()` & `plot_parameters()`: Utilities for handling batch simulations and generating standardized plots.

* **`project_field_h5_files/`** (Generated) Directory where simulation results and `.h5` data files are stored.

## 🛠️ Installation & Usage

### Prerequisites
You will need Python installed along with the following libraries:
* `py-tdgl`
* `numpy`
* `matplotlib`
* `h5py`
* `scipy`

### Running Locally
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/AlejoProjects/diode_effect.git](https://github.com/AlejoProjects/diode_effect.git)
    cd diode_effect
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    # Or manually: pip install tdgl numpy matplotlib h5py
    ```
3.  **Launch Jupyter:**
    ```bash
    jupyter notebook proyecto_electro2.ipynb
    ```

### Running on Google Colab
The notebook is optimized for Colab.
1.  Upload `proyecto_electro2.ipynb` to Colab.
2.  Follow the **"Collab execution"** instructions in the notebook's markdown cells:
    * Run the installation cells for `py-tdgl`.
    * Copy the content of `default_functions.py` into the dedicated cell (or upload the script to the runtime).

## 📊 Example Visualization

*Visualize the superconducting order parameter dynamics:*

> *[Insert a GIF or Image of your order parameter animation here]*

## 🤝 Contributing

Contributions are welcome! If you have ideas for new geometries, optimization techniques, or analysis tools:
1.  Fork the project.
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

*Created by [AlejoProjects](https://github.com/AlejoProjects)*
