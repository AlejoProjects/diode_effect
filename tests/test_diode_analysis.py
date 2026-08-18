import unittest

import numpy as np

from diode_analysis import (
    critical_current_at_voltage,
    differential_resistance,
    diode_efficiency,
    diode_metrics_from_threshold,
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


if __name__ == "__main__":
    unittest.main()
