import unittest

import numpy as np

from diode_analysis import (
    analyze_diode_branches,
    critical_current_at_voltage,
    differential_resistance,
    diode_efficiency,
    diode_metrics_from_threshold,
    plan_refined_sampling,
    transition_diagnostics,
    shared_voltage_criterion,
)


class DiodeAnalysisTests(unittest.TestCase):
    def test_differential_resistance_supports_descending_current(self):
        currents = np.array([0.0, -1.0, -2.0, -3.0])
        voltages = 2.0 * currents
        np.testing.assert_allclose(differential_resistance(currents, voltages), 2.0)

    def test_threshold_crossing_is_interpolated_and_signed(self):
        positive = critical_current_at_voltage(
            [0.0, 1.0, 2.0], [0.0, 0.2, 1.2], 0.7
        )
        negative = critical_current_at_voltage(
            [0.0, -1.0, -2.0], [0.0, -0.2, -1.2], 0.7
        )
        self.assertAlmostEqual(positive, 1.5)
        self.assertAlmostEqual(negative, -1.5)

    def test_diode_efficiency_handles_scalars_arrays_and_zero(self):
        self.assertAlmostEqual(diode_efficiency(12.0, -8.0), 0.2)
        self.assertEqual(diode_efficiency(0.0, 0.0), 0.0)
        np.testing.assert_allclose(
            diode_efficiency([12.0, 5.0], [-8.0, -5.0]), [0.2, 0.0]
        )

    def test_pair_metrics(self):
        metrics = diode_metrics_from_threshold(
            [0.0, 1.0, 2.0],
            [0.0, 0.4, 1.4],
            [0.0, -1.0, -2.0],
            [0.0, -0.6, -1.6],
            0.9,
        )
        self.assertAlmostEqual(metrics.ic_positive, 1.5)
        self.assertAlmostEqual(metrics.ic_negative, -1.3)
        self.assertAlmostEqual(metrics.efficiency, 1 / 14)

    def test_invalid_iv_data_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "strictly monotonic"):
            differential_resistance([0, 1, 1], [0, 1, 2])
        with self.assertRaisesRegex(ValueError, "never crosses"):
            critical_current_at_voltage([0, 1], [0, 0.1], 1.0)

    def test_transition_requires_consecutive_points(self):
        current = np.arange(7, dtype=float)
        voltage = np.array([0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 2.0])
        critical = critical_current_at_voltage(
            current, voltage, 1.0, min_consecutive=3
        )
        self.assertAlmostEqual(critical, 3.5)

    def test_missing_transition_returns_diagnostic_instead_of_failing(self):
        diagnostic = transition_diagnostics(
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0, 0.0],
            0.01,
        )
        self.assertFalse(diagnostic.reached_threshold)
        self.assertIsNone(diagnostic.critical_current)
        self.assertIn("Increase |I|max", diagnostic.message)

    def test_paired_diode_analysis_uses_one_shared_threshold(self):
        positive_current = np.linspace(0.0, 10.0, 101)
        negative_current = np.linspace(0.0, -10.0, 101)
        positive_voltage = np.maximum(positive_current - 5.0, 0.0) * 0.1
        negative_voltage = -np.maximum(np.abs(negative_current) - 6.0, 0.0) * 0.1
        result = analyze_diode_branches(
            positive_current,
            positive_voltage,
            negative_current,
            negative_voltage,
        )
        self.assertTrue(result.is_valid)
        self.assertEqual(
            result.positive.voltage_threshold,
            result.negative.voltage_threshold,
        )
        self.assertLess(result.efficiency, 0.0)

    def test_efficiency_is_stable_at_float_extremes(self):
        self.assertAlmostEqual(diode_efficiency(1e308, -5e307), 1 / 3)
        smallest = np.nextafter(0.0, 1.0)
        self.assertEqual(diode_efficiency(smallest, -smallest), 0.0)

    def test_empty_refinement_plan_keeps_the_coarse_grid(self):
        plan = plan_refined_sampling([], 161)
        self.assertEqual(plan["critical_steps"], 0)
        self.assertEqual(plan["normal_steps"], 161)
        self.assertIn("refinement was skipped", plan["message"])

    def test_refinement_plan_rejects_non_finite_boundaries(self):
        with self.assertRaisesRegex(ValueError, "total_steps"):
            plan_refined_sampling([1.0], np.inf)
        with self.assertRaisesRegex(ValueError, "critical_fraction"):
            plan_refined_sampling([1.0], 20, np.nan)

    def test_paired_analysis_rejects_wrong_branch_signs(self):
        with self.assertRaisesRegex(ValueError, "positive_currents"):
            analyze_diode_branches(
                [0.0, -1.0, -2.0],
                [0.0, 0.1, 0.2],
                [0.0, -1.0, -2.0],
                [0.0, -0.1, -0.2],
            )

    def test_paired_analysis_rejects_reversed_sweep_direction(self):
        with self.assertRaisesRegex(ValueError, "increase away from zero"):
            analyze_diode_branches(
                [2.0, 1.0, 0.0],
                [0.2, 0.1, 0.0],
                [0.0, -1.0, -2.0],
                [0.0, -0.1, -0.2],
            )

    def test_minimum_run_length_cannot_exceed_samples(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            transition_diagnostics(
                [0.0, 1.0, 2.0], [0.0, 0.1, 0.2], 0.1, min_consecutive=4
            )

    def test_current_difference_overflow_is_rejected(self):
        maximum = np.finfo(float).max
        with self.assertRaisesRegex(ValueError, "differences overflowed"):
            differential_resistance([-maximum, maximum], [0.0, 1.0])

    def test_extreme_noise_cannot_create_an_infinite_criterion(self):
        maximum = np.finfo(float).max
        with self.assertRaisesRegex(ValueError, "finite positive voltage criterion"):
            shared_voltage_criterion(
                [0.0, maximum, -maximum],
                [0.0, maximum, -maximum],
            )


if __name__ == "__main__":
    unittest.main()
