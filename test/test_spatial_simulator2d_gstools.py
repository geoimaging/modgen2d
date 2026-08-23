"""Tests for GSToolsSimulator."""

import numpy as np
from .testing_tools import unittest, TestCase
from modgen2d.spatial_simulator2d import GSToolsSimulator

class TestGSToolsSimulator(TestCase):
    def setUp(self):
        self.points = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    def test_zero_sigma_returns_exact_mean(self):
        result = GSToolsSimulator(2., 4., 100).simulate(self.points, [5, 3], 0)
        self.assertArrayEqual(result, 5 + 3 * self.points[:, 1])
    def test_seeded_results_are_reproducible(self):
        a = GSToolsSimulator(2., 4., 200, rng=np.random.default_rng(9))
        b = GSToolsSimulator(2., 4., 200, rng=np.random.default_rng(9))
        self.assertArrayAlmostEqual(a.simulate(self.points), b.simulate(self.points))
    def test_output_is_finite_and_correct_shape(self):
        result = GSToolsSimulator(2., 4., 100).simulate(self.points)
        self.assertEqual(result.shape, (4,)); self.assertTrue(np.all(np.isfinite(result)))
    def test_invalid_lengths_and_mode_count(self):
        for value in (0, -1, np.inf, np.nan):
            with self.assertRaises(ValueError): GSToolsSimulator(value, 1.)
        with self.assertRaises(TypeError): GSToolsSimulator(True, 1.)
        with self.assertRaises(TypeError): GSToolsSimulator(1., 1., 1.5)
        with self.assertRaises(ValueError): GSToolsSimulator(1., 1., 0)

if __name__ == '__main__':
    unittest.main()