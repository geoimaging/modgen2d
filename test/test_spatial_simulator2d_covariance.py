import numpy as np
from .testing_tools import unittest, TestCase
from modgen2d.spatial_simulator2d import CovarianceDecompositionSimulator, ConstantSimulator, SpatialSimulator2DAbstract, load_simulator_from_config
from modgen2d.general_functions import validate_processed_property_dict, check_for_zero_sigma

class TestSpatialSimulator2D(TestCase):
    
    def setUp(self):
        self.points = np.array([[0, 0], [1, 1], [2, 2], [3, 1], [0,3]])
        self.rng = np.random.default_rng(42)  
    
    def test_covariance_decomposition_simulate_shape(self):
        cov_sim = CovarianceDecompositionSimulator(theta_x=1.0, theta_z=1.0, rng=self.rng)
        result = cov_sim.simulate(self.points, mean=0, sigma=1)
        self.assertEqual(result.shape[0], self.points.shape[0])

    def test_covariance_decomposition_sigma_zero(self):
        cov_sim = CovarianceDecompositionSimulator(theta_x=1.0, theta_z=1.0, rng=self.rng)
        result = cov_sim.simulate(self.points, mean=[3, 2], sigma=0)
        expected = 3 + 2 * self.points[:, 1]
        self.assertArrayEqual(result, expected)
        
    def test_reproducibility_with_seed(self):
        cov_sim1 = CovarianceDecompositionSimulator(1.0, 1.0, rng=np.random.default_rng(123))
        cov_sim2 = CovarianceDecompositionSimulator(1.0, 1.0, rng=np.random.default_rng(123))
        result1 = cov_sim1.simulate(self.points, mean=0, sigma=1)
        result2 = cov_sim2.simulate(self.points, mean=0, sigma=1)
        self.assertArrayEqual(result1, result2)

    def test_covariance_decomposition_correlation_matrix(self):
        cov_sim = CovarianceDecompositionSimulator(theta_x=1.0, theta_z=1.0, rng=self.rng)
        L = cov_sim._compute_correlation_matrix(self.points)
        self.assertTrue(np.allclose(L, np.tril(L)))  # check lower-triangular
        eigvals = np.linalg.eigvalsh(L @ L.T)
        self.assertTrue(np.all(eigvals > 0))  # positive definite

    def test_zero_sigma_returns_mean_without_rng_draw(self):
        sim = CovarianceDecompositionSimulator(1., 1., rng=np.random.default_rng(5))
        before = sim.rng.bit_generator.state
        self.assertArrayEqual(sim.simulate(self.points, [3, 2], 0),
                              3 + 2 * self.points[:, 1])
        self.assertEqual(sim.rng.bit_generator.state, before)

    def test_factor_reconstructs_prescribed_correlation(self):
        sim = CovarianceDecompositionSimulator(2., 4.)
        factor = sim._compute_correlation_matrix(self.points)
        dx = np.abs(self.points[:, 0, None] - self.points[:, 0])
        dz = np.abs(self.points[:, 1, None] - self.points[:, 1])
        expected = np.exp(-2 * dx / 2.) * np.exp(-2 * dz / 4.)
        self.assertArrayAlmostEqual(factor @ factor.T, expected)
        self.assertTrue(np.allclose(factor, np.tril(factor)))
    
    def test_single_point_simulation(self):
        sim = CovarianceDecompositionSimulator(1.0, 1.0, rng=self.rng)
        point = np.array([[1.0, 2.0]])
        result = sim.simulate(point, mean=[2, 1], sigma=1)
        self.assertEqual(result.shape, (1,))

    def test_invalid_theta_types(self):
        with self.assertRaises(TypeError):
            CovarianceDecompositionSimulator("x", 1.0, rng=self.rng)

        with self.assertRaises(TypeError):
            CovarianceDecompositionSimulator(1.0, "z", rng=self.rng)

    def test_invalid_lengths(self):
        for value in (0, -1, np.inf, np.nan):
            with self.assertRaises(ValueError): 
                CovarianceDecompositionSimulator(value, 1.)
        for value in ('x', True):
            with self.assertRaises(TypeError): 
                CovarianceDecompositionSimulator(value, 1.)

    def test_get_config_and_from_config(self):
        cov_sim = CovarianceDecompositionSimulator(
            1.0,
            2.0,
            simulated_val_for_ignored_lit_property=-999,
            rng=np.random.default_rng(42),
        )

        config = cov_sim.get_config
        recreated = load_simulator_from_config(config)

        params_expected = {'theta_x': 1.0,
                           'theta_z': 2.0,
                            }
        self.assertEqual(recreated.params, params_expected)
        self.assertEqual(recreated.simulated_val_for_ignored_lit_property, -999)
        self.assertEqual(config["simulator_type_name"], "CovarianceDecompositionSimulator")
        self.assertEqual(recreated.allow_simulation, True)
    
if __name__ == "__main__":
    unittest.main()
