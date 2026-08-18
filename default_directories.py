"""Filesystem helpers for simulation outputs.

The functions in this module deliberately keep deletion opt-in. Constructing the
default directory tree never removes results from an earlier run.
"""

from pathlib import Path
import shutil
import tempfile

import numpy as np


DIRECTORY_NAMES = (
    "1_df_fixed_current_plots",
    "2_zero_currents_field_plots",
    "3_magnetization_plots",
    "4_varying_currents_plots",
)


def check_create_dir(directory):
    """Create *directory* and its parents, returning its normalized path."""
    path = Path(directory).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def create_default_directories(source="./figures", suffix=""):
    """Create and return the standard simulation output directories.

    ``pathlib`` joining is used so callers no longer need to include a trailing
    slash in ``source``. The return order is retained for notebook compatibility.
    """
    root = Path(source).expanduser()
    root.mkdir(parents=True, exist_ok=True)

    section1_dir, section2_dir, section3_dir, section4_dir = (
        root / f"{suffix}{name}" for name in DIRECTORY_NAMES
    )
    subsection21_dir = section2_dir / "21_varying_field_zero_current"
    subsection22_dir = section2_dir / "22_varying_heights_zero_current"
    subsection41_dir = section4_dir / "41_constant_field"
    subsection42_dir = section4_dir / "42_constant_field_different_heights"
    subsection43_dir = section4_dir / "43_different_fields"

    paths = (
        section1_dir,
        section2_dir,
        section3_dir,
        section4_dir,
        subsection21_dir,
        subsection22_dir,
        subsection41_dir,
        subsection42_dir,
        subsection43_dir,
    )
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
    return tuple(str(path) for path in paths)


def _validate_cleanup_target(directory):
    """Reject cleanup targets that are too broad to be safe."""
    path = Path(directory).expanduser().resolve()
    protected = {Path(path.anchor), Path.home().resolve(), Path.cwd().resolve()}
    if path in protected:
        raise ValueError(f"Refusing to clean protected directory: {path}")
    return path


def clean_source(source="./figures"):
    """Remove and recreate the complete default output tree."""
    root = _validate_cleanup_target(source)
    if root.exists():
        shutil.rmtree(root)
    return create_default_directories(root)


def clean_directory(directory):
    """Remove all contents from a directory and recreate the directory."""
    path = _validate_cleanup_target(directory)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def clean_static(source="./figures"):
    """Clear the static-current and zero-current field output sections."""
    paths = create_default_directories(source)
    for path in (paths[0], paths[4], paths[5]):
        clean_directory(path)


def save_data(data, file_path, column_titles):
    """Save two equally sized one-dimensional arrays as a text table."""
    if len(data) != 2:
        raise ValueError("data must contain exactly two columns")
    column_1 = np.asarray(data[0], dtype=float)
    column_2 = np.asarray(data[1], dtype=float)
    if column_1.ndim != 1 or column_2.ndim != 1:
        raise ValueError("both data columns must be one-dimensional")
    if column_1.size != column_2.size:
        raise ValueError("data columns must have the same number of values")
    if column_1.size == 0:
        raise ValueError("data columns must not be empty")
    if not np.all(np.isfinite(column_1)) or not np.all(np.isfinite(column_2)):
        raise ValueError("data columns must contain only finite values")

    path = Path(file_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.column_stack((column_1, column_2)), header=column_titles)


def clean_temp_files(prefix="electro2_", *, dry_run=False):
    """Remove project-owned directories from the system temporary directory.

    Only directories whose names start with ``prefix`` are considered. Set
    ``dry_run=True`` to inspect the targets without deleting them.

    Returns:
        A summary dictionary containing ``matched``, ``deleted``, and ``errors``.
    """
    if not prefix or prefix in {".", ".."}:
        raise ValueError("prefix must be a non-empty, specific directory prefix")

    temp_root = Path(tempfile.gettempdir()).resolve()
    matched = []
    deleted = []
    errors = []

    for path in temp_root.iterdir():
        if not path.name.startswith(prefix) or not path.is_dir():
            continue
        resolved = path.resolve()
        if resolved.parent != temp_root:
            errors.append((str(path), "target is outside the temporary directory"))
            continue
        matched.append(str(resolved))
        if dry_run:
            continue
        try:
            shutil.rmtree(resolved)
            deleted.append(str(resolved))
        except OSError as exc:
            errors.append((str(resolved), str(exc)))

    return {"matched": matched, "deleted": deleted, "errors": errors}


def load_two_columns(filepath, col1_name=None, col2_name=None, both=False):
    """Load a two-column text table written by :func:`save_data`.

    ``col1_name`` and ``col2_name`` are retained for API compatibility; the text
    format does not store structured column names beyond its header.
    """
    del col1_name, col2_name
    data = np.loadtxt(filepath, comments="#", ndmin=2)
    if data.size == 0 or data.shape[0] == 0:
        raise ValueError(f"No numeric rows found in {filepath!s}")
    if data.shape[1] < 2:
        raise ValueError(f"Expected at least two columns in {filepath!s}")
    if not np.all(np.isfinite(data[:, :2])):
        raise ValueError(f"Non-finite values found in {filepath!s}")
    if both:
        return data[:, 0], data[:, 1]
    return data[:, 1]


def load_current_sweep(directory):
    """Load and validate one saved voltage/resistance current sweep."""
    directory = Path(directory)
    voltage_path = directory / "voltage_vs_current.txt"
    resistance_path = directory / "resistance_vs_current.txt"
    currents, voltages = load_two_columns(voltage_path, both=True)
    resistance_currents, resistances = load_two_columns(resistance_path, both=True)
    if currents.shape != resistance_currents.shape or not np.allclose(
        currents, resistance_currents, rtol=1e-12, atol=1e-15
    ):
        raise ValueError(
            f"current axes do not match between {voltage_path} and {resistance_path}"
        )
    if currents.size > 1:
        with np.errstate(over="ignore", invalid="ignore"):
            differences = np.diff(currents)
        if not np.all(np.isfinite(differences)) or not (
            np.all(differences > 0) or np.all(differences < 0)
        ):
            raise ValueError(
                f"saved sweep in {directory} has a non-monotonic current axis"
            )
        current_scale = max(1.0, float(np.max(np.abs(currents))))
        if np.min(np.abs(differences)) <= 8 * np.finfo(float).eps * current_scale:
            raise ValueError(
                f"saved sweep in {directory} has numerically indistinguishable currents"
            )
    return currents, voltages, resistances


def height_directory_name(height):
    """Return a stable, round-trip-safe directory name for a height increment."""
    height = float(height)
    if not np.isfinite(height):
        raise ValueError("height must be finite")
    if height == 0:
        height = 0.0
    tag = np.format_float_positional(height, unique=True, trim="-")
    return f"dy_{tag}"


def load_height_sweeps(directory, heights):
    """Load ``dy_<height>`` sweeps and require a common current axis."""
    heights = np.asarray(heights, dtype=float)
    if heights.ndim != 1 or heights.size == 0 or not np.all(np.isfinite(heights)):
        raise ValueError("heights must be a non-empty finite one-dimensional array")
    tags = [height_directory_name(height) for height in heights]
    if len(set(tags)) != len(tags):
        raise ValueError("heights must not contain duplicates")
    current_axis = None
    voltage_series = []
    resistance_series = []
    for height, tag in zip(heights, tags):
        sweep_directory = Path(directory) / tag
        currents, voltages, resistances = load_current_sweep(sweep_directory)
        if current_axis is None:
            current_axis = currents
        elif currents.shape != current_axis.shape or not np.allclose(
            currents, current_axis, rtol=1e-12, atol=1e-15
        ):
            raise ValueError(f"current axis differs for height increment {height:g}")
        voltage_series.append(voltages)
        resistance_series.append(resistances)
    return current_axis, voltage_series, resistance_series
