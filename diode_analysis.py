"""Numerical analysis utilities for superconducting diode I-V curves.

This module has only a NumPy dependency, so saved simulation data can be analysed
without importing the much heavier pyTDGL solver stack.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _positive_integer(value, name):
    """Normalize a positive-integer option and reject booleans/non-finite values."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if not np.isfinite(numeric) or not numeric.is_integer() or numeric < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(numeric)


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
    with np.errstate(over="ignore", invalid="ignore"):
        differences = np.diff(currents)
    if not np.all(np.isfinite(differences)):
        raise ValueError("adjacent current differences overflowed")
    if np.any(differences == 0) or not (
        np.all(differences > 0) or np.all(differences < 0)
    ):
        raise ValueError("currents must be strictly monotonic with no duplicates")
    current_scale = max(1.0, float(np.max(np.abs(currents))))
    minimum_spacing = 8 * np.finfo(float).eps * current_scale
    if np.min(np.abs(differences)) <= minimum_spacing:
        raise ValueError(
            "adjacent current values are too close for stable numerical derivatives"
        )
    return currents, voltages


def plan_refined_sampling(critical_currents, total_steps, critical_fraction=0.6):
    """Return safe per-region sample counts for an optional refined sweep.

    An empty transition list is a valid diagnostic outcome: all samples remain in
    the coarse grid and no division is attempted. The returned counts are per
    critical region and per normal interval, matching the legacy notebook API.
    """
    critical_currents = np.asarray(critical_currents, dtype=float)
    if critical_currents.ndim != 1 or not np.all(np.isfinite(critical_currents)):
        raise ValueError("critical_currents must be a finite one-dimensional array")
    try:
        total_steps_value = float(total_steps)
        fraction = float(critical_fraction)
    except (TypeError, ValueError) as exc:
        raise ValueError("total_steps and critical_fraction must be numeric") from exc
    if (
        not np.isfinite(total_steps_value)
        or not total_steps_value.is_integer()
        or total_steps_value < 2
    ):
        raise ValueError("total_steps must be an integer greater than one")
    if not np.isfinite(fraction) or not 0 <= fraction < 1:
        raise ValueError("critical_fraction must be finite and in [0, 1)")

    total_steps_int = int(total_steps_value)
    count = critical_currents.size
    if count == 0:
        return {
            "critical_steps": 0,
            "normal_steps": total_steps_int,
            "message": (
                "No transition was detected, so refinement was skipped. "
                "Increase |I|max or change the applied field before refining."
            ),
        }

    critical_steps = max(1, int(round(total_steps_int * fraction / count)))
    normal_steps = max(
        1, int(round(total_steps_int * (1 - fraction) / (count + 1)))
    )
    return {
        "critical_steps": critical_steps,
        "normal_steps": normal_steps,
        "message": f"Allocated samples around {count} detected transition(s).",
    }


def differential_resistance(currents, voltages):
    """Return the numerical differential resistance ``dV/dI``."""
    currents, voltages = validate_iv_data(currents, voltages)
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            resistance = np.gradient(voltages, currents)
    except FloatingPointError as exc:
        raise ValueError("dV/dI overflowed; check current spacing and voltage scale") from exc
    if not np.all(np.isfinite(resistance)):
        raise ValueError("dV/dI contains non-finite values")
    return resistance


def critical_current_at_voltage(
    currents, voltages, voltage_threshold, *, min_consecutive=1
):
    """Return the first signed current crossing ``abs(V) = voltage_threshold``.

    Linear interpolation between adjacent samples reduces grid-quantization error.
    """
    currents, voltages = validate_iv_data(currents, voltages)
    threshold = float(voltage_threshold)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("voltage_threshold must be finite and positive")
    min_consecutive = _positive_integer(min_consecutive, "min_consecutive")
    if min_consecutive > currents.size:
        raise ValueError("min_consecutive cannot exceed the number of samples")

    magnitude = np.abs(voltages)
    above = magnitude >= threshold
    if min_consecutive > 1:
        kernel = np.ones(min_consecutive, dtype=int)
        sustained = np.convolve(above.astype(int), kernel, mode="valid") == min_consecutive
        sustained_starts = np.flatnonzero(sustained)
        crossing_candidates = sustained_starts[sustained_starts > 0] - 1
    else:
        crossing_candidates = np.flatnonzero((~above[:-1]) & above[1:])
    crossings = crossing_candidates[
        magnitude[crossing_candidates] < threshold
    ]
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
    # Normalize before adding/subtracting to avoid overflow for very large values
    # and underflow for two finite subnormal values.
    scale = np.maximum(positive, negative)
    scaled_positive = np.divide(
        positive, scale, out=np.zeros_like(positive), where=scale != 0
    )
    scaled_negative = np.divide(
        negative, scale, out=np.zeros_like(negative), where=scale != 0
    )
    denominator = scaled_positive + scaled_negative
    efficiency = np.divide(
        scaled_positive - scaled_negative,
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


@dataclass(frozen=True)
class TransitionDiagnostics:
    """Result of looking for one sustained resistive transition."""

    critical_current: Optional[float]
    voltage_threshold: float
    maximum_voltage: float
    noise_floor: float
    reached_threshold: bool
    message: str


@dataclass(frozen=True)
class DiodeSweepDiagnostics:
    """Paired positive/negative transition result at a shared criterion."""

    positive: TransitionDiagnostics
    negative: TransitionDiagnostics
    efficiency: Optional[float]

    @property
    def is_valid(self):
        return self.efficiency is not None


def _noise_floor(voltages, baseline_points=None):
    """Estimate the low-current voltage noise using a robust MAD estimator."""
    voltages = finite_1d(voltages, "voltages", min_size=3)
    if baseline_points is None:
        baseline_points = max(3, int(np.ceil(0.1 * voltages.size)))
    baseline_points = min(int(baseline_points), voltages.size)
    baseline = voltages[:baseline_points]
    scale = float(np.max(np.abs(baseline)))
    if scale == 0:
        return float(np.finfo(float).eps)
    normalized = baseline / scale
    median = np.median(normalized)
    mad = float(np.median(np.abs(normalized - median)))
    # Work in normalized coordinates and clamp only at the representable limit.
    noise = min(np.finfo(float).max, 1.4826 * mad * scale)
    return float(max(noise, np.finfo(float).eps))


def transition_diagnostics(
    currents,
    voltages,
    voltage_threshold,
    *,
    min_consecutive=3,
):
    """Return a non-throwing diagnostic for a threshold-defined transition."""
    min_consecutive = _positive_integer(min_consecutive, "min_consecutive")
    currents, voltages = validate_iv_data(currents, voltages, min_size=3)
    if min_consecutive > currents.size:
        raise ValueError("min_consecutive cannot exceed the number of samples")
    threshold = float(voltage_threshold)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("voltage_threshold must be finite and positive")
    maximum_voltage = float(np.max(np.abs(voltages)))
    noise_floor = _noise_floor(voltages)
    try:
        critical_current = critical_current_at_voltage(
            currents,
            voltages,
            threshold,
            min_consecutive=min_consecutive,
        )
    except ValueError as exc:
        return TransitionDiagnostics(
            critical_current=None,
            voltage_threshold=threshold,
            maximum_voltage=maximum_voltage,
            noise_floor=noise_floor,
            reached_threshold=False,
            message=(
                f"No sustained transition at |V|={threshold:g}; "
                f"max |V|={maximum_voltage:g}. Increase |I|max, adjust the field, "
                "or lower the criterion only if it remains above the noise floor."
            ),
        )
    return TransitionDiagnostics(
        critical_current=critical_current,
        voltage_threshold=threshold,
        maximum_voltage=maximum_voltage,
        noise_floor=noise_floor,
        reached_threshold=True,
        message=f"Transition detected at I={critical_current:g}.",
    )


def shared_voltage_criterion(
    positive_voltages,
    negative_voltages,
    *,
    relative_level=0.05,
    noise_multiplier=6.0,
):
    """Choose one noise-aware voltage criterion for both current directions."""
    positive = finite_1d(positive_voltages, "positive_voltages", min_size=3)
    negative = finite_1d(negative_voltages, "negative_voltages", min_size=3)
    try:
        relative_level = float(relative_level)
        noise_multiplier = float(noise_multiplier)
    except (TypeError, ValueError) as exc:
        raise ValueError("criterion controls must be numeric") from exc
    if not np.isfinite(relative_level) or not 0 < relative_level < 1:
        raise ValueError("relative_level must be between zero and one")
    if not np.isfinite(noise_multiplier) or noise_multiplier <= 0:
        raise ValueError("noise_multiplier must be finite and positive")
    common_peak = min(float(np.max(np.abs(positive))), float(np.max(np.abs(negative))))
    try:
        noise_limit = noise_multiplier * max(
            _noise_floor(positive), _noise_floor(negative)
        )
        relative_limit = relative_level * common_peak
    except OverflowError as exc:
        raise ValueError("voltage scale overflowed while choosing a criterion") from exc
    threshold = max(noise_limit, relative_limit)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("could not determine a finite positive voltage criterion")
    return float(threshold)


def analyze_diode_branches(
    positive_currents,
    positive_voltages,
    negative_currents,
    negative_voltages,
    voltage_threshold=None,
    *,
    min_consecutive=3,
):
    """Analyze both branches without manufacturing a result when one is missing."""
    positive_currents, positive_voltages = validate_iv_data(
        positive_currents, positive_voltages, min_size=3
    )
    negative_currents, negative_voltages = validate_iv_data(
        negative_currents, negative_voltages, min_size=3
    )
    current_scale = max(
        1.0,
        float(np.max(np.abs(positive_currents))),
        float(np.max(np.abs(negative_currents))),
    )
    sign_tolerance = 8 * np.finfo(float).eps * current_scale
    if np.any(positive_currents < -sign_tolerance):
        raise ValueError("positive_currents must contain only non-negative values")
    if np.any(negative_currents > sign_tolerance):
        raise ValueError("negative_currents must contain only non-positive values")
    if not np.all(np.diff(positive_currents) > 0):
        raise ValueError("positive_currents must increase away from zero")
    if not np.all(np.diff(negative_currents) < 0):
        raise ValueError("negative_currents must decrease away from zero")
    if voltage_threshold is None:
        voltage_threshold = shared_voltage_criterion(
            positive_voltages, negative_voltages
        )
    positive = transition_diagnostics(
        positive_currents,
        positive_voltages,
        voltage_threshold,
        min_consecutive=min_consecutive,
    )
    negative = transition_diagnostics(
        negative_currents,
        negative_voltages,
        voltage_threshold,
        min_consecutive=min_consecutive,
    )
    efficiency = None
    if positive.reached_threshold and negative.reached_threshold:
        if positive.critical_current < 0 or negative.critical_current > 0:
            raise ValueError("positive and negative branches have unexpected current signs")
        efficiency = diode_efficiency(
            positive.critical_current, negative.critical_current
        )
    return DiodeSweepDiagnostics(
        positive=positive,
        negative=negative,
        efficiency=efficiency,
    )


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
