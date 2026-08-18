from pathlib import Path
import tempfile
import unittest

import numpy as np

import default_directories as directories


class DirectoryHelperTests(unittest.TestCase):
    def temporary_directory(self):
        # Keep test artifacts inside the writable project workspace. This also
        # makes the suite work in restricted CI/sandbox environments.
        return tempfile.TemporaryDirectory(dir=Path.cwd())

    def test_tree_uses_source_as_a_directory_without_trailing_slash(self):
        with self.temporary_directory() as temporary_directory:
            root = Path(temporary_directory) / "figures"
            paths = directories.create_default_directories(root)
            self.assertEqual(len(paths), 9)
            self.assertTrue(all(Path(path).is_dir() for path in paths))
            self.assertTrue(all(Path(path).is_relative_to(root) for path in paths))

    def test_single_row_round_trip(self):
        with self.temporary_directory() as temporary_directory:
            path = Path(temporary_directory) / "nested" / "data.txt"
            directories.save_data(([1.0], [2.0]), path, "x y")
            first, second = directories.load_two_columns(path, both=True)
            np.testing.assert_allclose(first, [1.0])
            np.testing.assert_allclose(second, [2.0])

    def test_mismatched_columns_are_rejected(self):
        with self.temporary_directory() as temporary_directory:
            path = Path(temporary_directory) / "data.txt"
            with self.assertRaisesRegex(ValueError, "same number"):
                directories.save_data(([1.0], [2.0, 3.0]), path, "x y")

    def test_empty_and_non_finite_data_are_rejected(self):
        with self.temporary_directory() as temporary_directory:
            path = Path(temporary_directory) / "data.txt"
            with self.assertRaisesRegex(ValueError, "must not be empty"):
                directories.save_data(([], []), path, "x y")
            with self.assertRaisesRegex(ValueError, "finite"):
                directories.save_data(([0.0], [np.inf]), path, "x y")

    def test_cleanup_refuses_workspace_root(self):
        with self.assertRaisesRegex(ValueError, "protected"):
            directories.clean_directory(Path.cwd())

    def test_height_sweeps_are_loaded_on_one_validated_axis(self):
        with self.temporary_directory() as temporary_directory:
            root = Path(temporary_directory) / "sweeps"
            currents = np.array([0.0, 1.0, 2.0])
            for height in (0.0, 1.0):
                sweep = root / f"dy_{height:g}"
                directories.save_data(
                    (currents, currents + height),
                    sweep / "voltage_vs_current.txt",
                    "I V",
                )
                directories.save_data(
                    (currents, np.ones_like(currents)),
                    sweep / "resistance_vs_current.txt",
                    "I dV/dI",
                )
            axis, voltages, resistances = directories.load_height_sweeps(
                root, [0.0, 1.0]
            )
            np.testing.assert_allclose(axis, currents)
            np.testing.assert_allclose(voltages[1], currents + 1.0)
            np.testing.assert_allclose(resistances[0], 1.0)

    def test_current_sweep_rejects_mismatched_saved_axes(self):
        with self.temporary_directory() as temporary_directory:
            root = Path(temporary_directory) / "sweep"
            directories.save_data(
                ([0.0, 1.0], [0.0, 0.1]),
                root / "voltage_vs_current.txt",
                "I V",
            )
            directories.save_data(
                ([0.0, 2.0], [0.0, 0.1]),
                root / "resistance_vs_current.txt",
                "I dV/dI",
            )
            with self.assertRaisesRegex(ValueError, "current axes do not match"):
                directories.load_current_sweep(root)


if __name__ == "__main__":
    unittest.main()
