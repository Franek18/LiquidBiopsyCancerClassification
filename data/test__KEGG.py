import unittest
import _KEGG as k
import numpy as np


class TestPreprocessing(unittest.TestCase):
    def permutate_signals_rand(self):
        np.random.seed(555)  # to avoid flaky tests
        self.assertTrue(np.any(
            k.permutate_signals(100) !=
            k.permutate_signals(100)
        ))

    def test_permutate_signals_consistent(self):
        self.assertTrue(np.all(
            k.permutate_signals(100, 123) ==
            k.permutate_signals(100, 123)
        ))

    def test_square_to_flat(self):
        self.assertTrue(np.all(
            np.array([[1, 2, 3, 4], [10, 20, 30, 40]]) ==
            k.square_matrices_to_flat(np.array([[[1, 2], [3, 4]], [[10, 20], [30, 40]]]))
        ))

    def test_flat_to_square_easy(self):
        self.assertTrue(np.all(
            np.array([[[1, 2], [3, 4]], [[10, 20], [30, 40]]]) ==
            k.flat_matrices_to_square(np.array([[1, 2, 3, 4], [10, 20, 30, 40]]))
        ))

    def test_flat_to_square_fill(self):
        self.assertTrue(np.all(
            np.array([[[1, 2], [3, 0]], [[10, 20], [30, 0]]]) ==
            k.flat_matrices_to_square(np.array([[1, 2, 3], [10, 20, 30]]))
        ))

    def test_reduce(self):
        x = np.array([[1, 2, 3, 4], [1.5, 2, 30, 4.0003]])
        stds = k.calculate_stds(x)
        self.assertTrue(np.all(
            np.array([[1, 3, 4], [1.5, 30, 4.0003]]) ==
            k.reduce_matrices(x, stds)
        ))


if __name__ == '__main__':
    unittest.main()
