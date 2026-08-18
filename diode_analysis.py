"""Numerical analysis utilities for superconducting diode I-V curves.

This module has only a NumPy dependency, so saved simulation data can be analysed
without importing the much heavier pyTDGL solver stack.
"""

from dataclasses import dataclass

import numpy as np


def finite_1d(values, name, *, min_size=1):
    """Return *values* as a validated one-dimensional float array."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size < min_size:
        raise ValueError(f"{name} must contain at least {min_size} values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def validate_iv_data(currents, voltages, *, min_size=2):
    """Validate a sampled I-V curve and return NumPy arrays."""
    currents = finite_1d(currents, "currents", min_size=min_size)
    voltages = finite_1d(voltages, "voltages", min_size=min_size)
    if currents.shape != voltages.shape:
        raise ValueError("currents and voltages must have the same shape")
    differences = np.diff(currents)
    if np.any(differences == 0) or not (
        np.all(differences > 0) or np.all(differences < 0)
    ):
        raise ValueError("currents must be strictly monotonic with no duplicates")
    return currents, voltages


def differential_resistance(currents, voltages):
    """Return the numerical differential resistance ``dV/dI``."""
    currents, voltages = validate_iv_data(currents, voltages)
    return np.gradient(voltages, currents)


def critical_current_at_voltage(currents, voltages, voltage_threshold):
    """Return the first signed current crossing ``abs(V) = voltage_threshold``.

    Linear interpolation between adjacent samples reduces grid-quantization error.
    """
    currents, voltages = validate_iv_data(currents, voltages)
    threshold = float(voltage_threshold)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("voltage_threshold must be finite and positive")

    magnitude = np.abs(voltages)
    crossings = np.flatnonzero(
        (magnitude[:-1] < threshold) & (magnitude[1:] >= threshold)
    )
    if crossings.size == 0:
        raise ValueError("the I-V curve never crosses the voltage threshold")
    index = int(crossings[0])
    v0, v1 = magnitude[index:index + 2]
    i0, i1 = currents[index:index + 2]
    fraction = (threshold - v0) / (v1 - v0)
    return float(i0 + fraction * (i1 - i0))


def diode_efficiency(ic_positive, ic_negative):
    """Return ``eta=(|Ic+|-|Ic-|)/(|Ic+|+|Ic-|)`` in the range [-1, 1]."""
    positive = np.abs(np.asarray(ic_positive, dtype=float))
    negative = np.abs(np.asarray(ic_negative, dtype=float))
    if positive.shape != negative.shape:
        raise ValueError(
            f"critical-current shapes must match: {positive.shape} != {negative.shape}"
        )
    if not np.all(np.isfinite(positive)) or not np.all(np.isfinite(negative)):
        raise ValueError("critical currents must be finite")
    denominator = positive + negative
    efficiency = np.divide(
        positive - negative,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator != 0,
    )
    return float(efficiency) if efficiency.ndim == 0 else efficiency


@dataclass(frozen=True)
class DiodeMetrics:
    """Critical currents and diode efficiency at one voltage criterion."""

    ic_positive: float
    ic_negative: float
    efficiency: float


def diode_metrics_from_threshold(
    positive_currents,
    positive_voltages,
    negative_currents,
    negative_voltages,
    voltage_threshold,
):
    """Analyse positive and negative I-V branches with one voltage criterion."""
    ic_positive = critical_current_at_voltage(
        positive_currents, positive_voltages, voltage_threshold
    )
    ic_negative = critical_current_at_voltage(
        negative_currents, negative_voltages, voltage_threshold
    )
    if ic_positive < 0 or ic_negative > 0:
        raise ValueError("positive and negative branches have unexpected current signs")
    return DiodeMetrics(
        ic_positive=ic_positive,
        ic_negative=ic_negative,
        efficiency=diode_efficiency(ic_positive, ic_negative),
    )
