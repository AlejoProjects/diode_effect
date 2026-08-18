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

    def test_cleanup_refuses_workspace_root(self):
        with self.assertRaisesRegex(ValueError, "protected"):
            directories.clean_directory(Path.cwd())


if __name__ == "__main__":
    unittest.main()
