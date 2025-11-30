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
### Important
* The parameter section must always be executed
* The create device can be modified to merge any kind of device. ( if you're using the default geometries be sure to add their imports from the tdgl)
* The title parameter for the plot_solution function or any tdgl.plot method is used for a modified function on the tdgl library  adding titles to the obtained figures (it's better to set title = None if you dont want to meedle with tdgl source code)
### Collab exectution
* If you're using colab a series of small steps are needed:
    * 1. use ctr+H on chrome or ctr+f on vscode(if you're using the new colab extension) to get to the find section, search for df. ,click the arrow left to the writting space and click replace all without typing anything on the replace writting space (if you want to, select the match the whole word button.)
      2. Go to the collab Section
      3. Run the commented step for the installation of the required libraries
      4. Copy the contents of the default_functions.py script into the functions cell
      5. copy the contents of the directories.py(yet to be implemented) script into the dir cell
      6. Remove the lines (if df is not removed):
        * import default_functions as df 
        * d = df.d  gamma = df.gamma
        * xi = df.xi           
        * london_lambda = df.london_lambda  
        * d = df.d               
        * gamma = df.gamma     
After that you'll be good to go
### Some advice
* Any of the functions used in this notebook can be used with any created device. It's just a matter of passing the device object to the function.
* To change the default simulation values you can edit the default_options and default_solutions functions to set the simulation times and any other parameters located on the default_functions.py script
* Each simulation section should run independently of the other sections as long as you've executed the parameters* section(customize the parameters as you see fit)
### Optional
* If your device supports it you can install cuda and run the script on your cpu adding the parameter on the tdgl.options located in the default_options function.
* If you're using a cluster you can divide the work needed to be done on the magnetization,varying currents or varying fields to save time.


## 🤝 Contributing

Contributions are welcome! If you have ideas for new geometries, optimization techniques, or analysis tools:
1.  Fork the project.
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

*Created by [AlejoProjects](https://github.com/AlejoProjects)*
