# Superconducting diode-effect simulator

This project models non-reciprocal transport in asymmetric superconducting thin
films with the generalized time-dependent Ginzburg–Landau (TDGL) equations. It
uses [pyTDGL](https://py-tdgl.readthedocs.io/en/latest/) for geometry, meshing,
time evolution, and solution post-processing.

The main experiment compares positive and negative current branches to estimate
the directional critical currents `Ic+` and `Ic-` and the diode efficiency

```text
eta = (|Ic+| - |Ic-|) / (|Ic+| + |Ic-|).
```

## Physical model and scope

The notebook constructs an asymmetric bridge from a central rectangle and two
unequal side arms. The geometry breaks spatial inversion symmetry. An applied
out-of-plane magnetic field breaks time-reversal symmetry; both ingredients are
normally required for an equilibrium superconducting diode response in this
type of conventional TDGL model.

pyTDGL evolves the normalized complex order parameter `psi`, scalar potential,
and sheet-current density in a two-dimensional thin film. The notebook explores:

- fixed-current solutions at multiple perpendicular magnetic fields;
- zero-current field sweeps and magnetic moment-derived magnetization;
- positive and negative I-V branches as the asymmetric arm height changes;
- order-parameter density, phase, current, vorticity, and scalar-potential plots;
- critical-current and diode-efficiency extraction.

Important limitations:

- This is a classical generalized TDGL simulation, not a microscopic theory of
  the diode effect.
- pyTDGL neglects magnetic screening by default. Magnetization results should not
  be interpreted as a fully self-consistent thermodynamic magnetization unless
  the chosen solver/device configuration includes the required screening model.
- Each current point is currently solved independently. That is appropriate for
  branch comparisons, but it does not trace metastable hysteresis by seeding each
  point from the previous solution.
- Critical currents depend on the voltage criterion, sampling density, relaxation
  time, mesh convergence, and noise treatment. A single reported value should be
  accompanied by those choices.
- pyTDGL can warn about malformed boundary Voronoi cells for this stepped geometry.
  The project rejects non-finite or non-positive mesh-cell areas, but that check
  does not replace a mesh/boundary-resampling convergence study before treating
  a small efficiency as physical.

## Repository layout

| Path | Purpose |
| --- | --- |
| `proyecto_electro2.ipynb` | End-to-end horizontal and vertical bridge studies. |
| `default_functions.py` | Device construction, terminal placement, TDGL sweeps, and visualization. |
| `diode_analysis.py` | Lightweight I-V analysis that only depends on NumPy. |
| `default_directories.py` | Safe output-tree, table save/load, and cleanup helpers. |
| `tests/` | Fast numerical and filesystem regression tests. |
| `requirements.txt` | Runtime and notebook dependencies. |
| `setup_venv.ps1` / `setup_venv.sh` | One-command virtual-environment setup. |

Generated figures and text tables are written under `figures/` or
`figures_vertical/` and are ignored by Git.

## Installation

Use Python 3.10–3.14; Python 3.12 is recommended. Although pyTDGL itself also
supports Python 3.9, the current JupyterLab dependency requires Python 3.10 or
newer. Do not install the project into the system Python environment.

### Windows PowerShell — automated

From the repository directory:

```powershell
.\setup_venv.ps1
.\.venv\Scripts\Activate.ps1
python -m jupyterlab proyecto_electro2.ipynb
```

If `python` is not the desired interpreter, pass its executable explicitly:

```powershell
.\setup_venv.ps1 -PythonCommand "C:\Path\To\Python312\python.exe"
```

If PowerShell prevents local scripts from running, use the manual commands below
instead of changing the machine-wide execution policy.

### macOS/Linux — automated

```bash
sh ./setup_venv.sh
. .venv/bin/activate
python -m jupyterlab proyecto_electro2.ipynb
```

Set `PYTHON_COMMAND=python3.12` before the script if more than one Python version
is installed.

### Manual installation

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or activate it on macOS/Linux:

```bash
. .venv/bin/activate
```

Then install everything through the virtual environment's interpreter:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install --requirement requirements.txt
python -m ipykernel install --user --name diode-effect --display-name "Python (diode-effect)"
python -m jupyterlab proyecto_electro2.ipynb
```

In JupyterLab, select **Python (diode-effect)** as the notebook kernel.

### Verify the environment

```bash
python -c "import tdgl, numpy, scipy, matplotlib, h5py, IPython; print('Environment OK')"
python -m unittest discover -s tests -v
```

### Google Colab

Upload or clone the complete repository, change into its directory, and run the
notebook's installation cell:

```python
%pip install --requirement requirements.txt
```

Restart the Colab runtime after the first install. There is no need to paste the
contents of the helper modules into notebook cells.

### Optional acceleration and animations

GPU packages are intentionally excluded from `requirements.txt`: the correct
CuPy build depends on the installed CUDA/ROCm toolkit and GPU driver. Follow the
[pyTDGL GPU installation guidance](https://py-tdgl.readthedocs.io/en/latest/installation.html#gpu-acceleration)
and enable `SolverOptions.gpu` only after confirming that CuPy detects the GPU.

FFmpeg is only needed when exporting animations. Install it with the operating
system package manager; it is not a Python library and therefore does not belong
in `requirements.txt`.

The stale embedded notebook outputs have been removed, so execute cells from the
top when starting a new study. TDGL sweeps can be expensive; begin with a small
number of current and field points to validate geometry and terminal placement.

## Typical workflow

1. Define material parameters and the bridge dimensions.
2. Draw the unmeshed device and inspect numbered boundary segments.
3. Select two terminal segments and verify the voltage-probe orientation,
   bounds-centered positions, and actual separation printed during meshing.
4. Run a coarse positive/negative current sweep at fixed magnetic field.
5. Check time traces to ensure the voltage averaging interval is in steady state.
6. Refine the mesh, time settings, and current grid around the transition.
7. Extract `Ic+`, `Ic-`, and `eta` with a documented voltage criterion.

The `Building` class is the preferred name. The old lowercase `building` name is
still available as a compatibility alias for existing notebook cells.

## Analysing saved I-V data

The lightweight analysis module can be used without importing pyTDGL:

```python
from default_directories import load_two_columns
from diode_analysis import diode_metrics_from_threshold

i_pos, v_pos = load_two_columns("positive/voltage_vs_current.txt", both=True)
i_neg, v_neg = load_two_columns("negative/voltage_vs_current.txt", both=True)

metrics = diode_metrics_from_threshold(
    i_pos,
    v_pos,
    i_neg,
    v_neg,
    voltage_threshold=0.01,  # in the saved pyTDGL voltage units
)
print(metrics)
```

For exploratory work, `Building.find_critical_currents()` locates peaks in
`abs(dV/dI)`. When a defensible voltage criterion is available,
`Building.estimate_critical_current()` or `diode_metrics_from_threshold()` is the
more reproducible choice.

Voltage is now stored with its sign. The first probe minus the second probe fixes
the voltage polarity. Use `absolute_voltage=True` in `current_application()` only
when reproducing legacy magnitude-only plots.

### Finding a credible diode-effect window

Section 4 starts with a coarse paired sweep at `B = 1 mT`, `|I| <= 30 µA`, and
161 samples. These are search starting points, not a claim that this exact device
must show a transition there. If either polarity does not cross the shared
voltage criterion, the analysis returns a diagnostic and `NaN` for that geometry
instead of dividing by zero or inventing an efficiency.

For a useful parameter search:

1. Run paired current branches at `B = ±0.5, ±1, ±1.5, ±2 mT` (and a zero-field
   control), increasing `|I|max` until both branches clearly reach the resistive
   state.
2. Use one voltage criterion above the low-current noise floor for every branch
   being compared. The notebook chooses a robust shared criterion automatically
   unless `voltage_criterion` is set explicitly.
3. Require at least three consecutive samples above the criterion; isolated
   voltage spikes are not accepted as a transition.
4. Look for nonzero `eta(B)` that approximately obeys `eta(-B) = -eta(B)` and
   approaches zero at `B = 0`. A field-even offset is a warning sign for probe,
   relaxation, meshing, or threshold bias.
5. Refine the current grid, mesh, and time averaging around the best candidate,
   then confirm that the result is stable to those numerical choices.

Use `Building.analyze_diode_branches()` for the non-throwing paired diagnostic
and `Building.plan_refined_sampling()` for safe refinement allocation. An empty
critical-region array now means “skip refinement and expand the coarse search”;
it is no longer treated as an exceptional divisor.

## Output conventions

- Current: the configured pyTDGL current unit (the notebook uses `uA`).
- Magnetic field: the configured field unit (the notebook uses `mT`).
- Length: the device length unit (the notebook uses `um`).
- Voltage: pyTDGL's dimensionless voltage scale `V0` unless converted separately.
- Magnetic moment: `uA * um**2` in the field-sweep helper.
- Magnetization: `uA / um**3`, computed as moment divided by film area and
  thickness.

For multi-height current sweeps, files are organized as
`<save_dir>/dy_<height>/voltage_vs_current.txt`. A single-height sweep writes
directly to `<save_dir>`.

## Verification

Run the fast local tests with:

```bash
python -m unittest discover -s tests -v
```

These tests cover signed positive/negative I-V analysis, scalar and vector diode
efficiency, threshold interpolation, invalid input handling, directory creation,
and table round trips. A full TDGL regression is intentionally not part of the
fast suite because it requires pyTDGL and a costly numerical solve.

For research-quality results, also perform mesh convergence, time-step/
relaxation convergence, current-grid refinement, positive/negative probe-order
checks, and field-reversal symmetry checks.

## References

- [pyTDGL installation](https://py-tdgl.readthedocs.io/en/latest/installation.html)
- [pyTDGL solver API](https://py-tdgl.readthedocs.io/en/latest/api/solver.html)
- [pyTDGL theoretical background](https://py-tdgl.readthedocs.io/en/latest/background.html)
- L. Bishop-Van Horn, *pyTDGL: Time-dependent Ginzburg-Landau in Python*,
  Computer Physics Communications 291, 108799 (2023).
