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
    column_1 = np.asarray(data[0])
    column_2 = np.asarray(data[1])
    if column_1.ndim != 1 or column_2.ndim != 1:
        raise ValueError("both data columns must be one-dimensional")
    if column_1.size != column_2.size:
        raise ValueError("data columns must have the same number of values")

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
    if data.shape[1] < 2:
        raise ValueError(f"Expected at least two columns in {filepath!s}")
    if both:
        return data[:, 0], data[:, 1]
    return data[:, 1]
