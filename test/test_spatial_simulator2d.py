from geomodgen2d.features_config import FeaturesConfig
import numpy as np
from testing_tools import unittest, TestCase
from geomodgen2d.spatial_simulator2d import CovarianceDecompositionSimulator, ConstantSimulator, SpatialSimulator2D, validate_processed_property_dict, check_for_zero_sigma

class TestSpatialSimulator2D(TestCase):
    
    def setUp(self):
        self.points = np.array([[0, 0], [1, 1], [2, 2], [3, 1], [0,3]])
        self.rng = np.random.default_rng(42)  
    
    def test_constant_simulator_output(self):
        cs = ConstantSimulator(self.rng)
        expected = 5 + 2 * self.points[:, 0]
        result = cs.simulate(self.points, mean=[5, 2])
        self.assertArrayEqual(result, expected)

        # test zero trend
        expected = np.full(self.points.shape[0], 10)
        result = cs.simulate(self.points, mean=10)
        self.assertArrayEqual(result, expected)

    def test_covariance_decomposition_simulate_shape(self):
        cov_sim = CovarianceDecompositionSimulator(theta_x=1.0, theta_z=1.0, rng=self.rng)
        result = cov_sim.simulate(self.points, mean=0, sigma=1)
        self.assertEqual(result.shape[0], self.points.shape[0])

    def test_covariance_decomposition_sigma_zero(self):
        cov_sim = CovarianceDecompositionSimulator(theta_x=1.0, theta_z=1.0, rng=self.rng)
        result = cov_sim.simulate(self.points, mean=[3, 2], sigma=0)
        expected = 3 + 2 * self.points[:, 0]
        self.assertArrayEqual(result, expected)
        
    def test_reproducibility_with_seed(self):
        cov_sim1 = CovarianceDecompositionSimulator(1.0, 1.0, rng=np.random.default_rng(123))
        cov_sim2 = CovarianceDecompositionSimulator(1.0, 1.0, rng=np.random.default_rng(123))
        result1 = cov_sim1.simulate(self.points, mean=0, sigma=1)
        result2 = cov_sim2.simulate(self.points, mean=0, sigma=1)
        self.assertArrayEqual(result1, result2)

    def test_covariance_decomposition_correlation_matrix(self):
        cov_sim = CovarianceDecompositionSimulator(theta_x=1.0, theta_z=1.0, rng=self.rng)
        L = cov_sim.compute_correlation_matrix(self.points)
        self.assertTrue(np.allclose(L, np.tril(L)))  # check lower-triangular
        eigvals = np.linalg.eigvalsh(L @ L.T)
        self.assertTrue(np.all(eigvals > 0))  # positive definite
    
    def test_get_means_am_bm(self):
        a, b = SpatialSimulator2D.get_means_am_bm(5)
        self.assertEqual(a, 5)
        self.assertEqual(b, 0)

        a, b = SpatialSimulator2D.get_means_am_bm([2, 3])
        self.assertEqual(a, 2)
        self.assertEqual(b, 3)

    def test_get_means_am_bm_invalid(self):
        with self.assertRaises(ValueError):
            SpatialSimulator2D.get_means_am_bm([1, 2, 3])

    def test_check_points_valid(self):
        pts = SpatialSimulator2D.check_points(self.points)
        self.assertArrayEqual(pts, self.points)

    def test_check_points_invalid_shape(self):
        with self.assertRaises(ValueError):
            SpatialSimulator2D.check_points([1, 2, 3])

    def test_check_points_non_numeric(self):
        with self.assertRaises(TypeError):
            SpatialSimulator2D.check_points([["a", "b"]])

    def test_single_point_simulation(self):
        sim = CovarianceDecompositionSimulator(1.0, 1.0, self.rng)
        point = np.array([[1.0, 2.0]])
        result = sim.simulate(point, mean=[2, 1], sigma=1)
        self.assertEqual(result.shape, (1,))

    def test_invalid_theta_types(self):
        with self.assertRaises(TypeError):
            CovarianceDecompositionSimulator("x", 1.0, self.rng)

        with self.assertRaises(TypeError):
            CovarianceDecompositionSimulator(1.0, "z", self.rng)

    def test_validate_valid(self):
        prop = {
            "L1": {
                "wet": {"mean": 10, "stdev/cov": 0.2, "stdev_type": "cov"},
                "dry": {"mean": 8, "stdev/cov": 1.5, "stdev_type": "stdev"},
            }
        }

        validated = validate_processed_property_dict(prop)

        self.assertEqual(validated["L1"]["wet"]["mean_bm"], 0.0)
        self.assertEqual(validated["L1"]["dry"]["mean_bm"], 0.0)

        prop = {
            "L2": {
                "both": {"mean": 5, "mean_bm": 1.2, "stdev/cov": 0.1, "stdev_type": "cov"}
            }
        }

        validated = validate_processed_property_dict(prop)
        self.assertEqual(validated["L2"]["both"]["mean"], 5)

    def test_validate_missing(self):
        prop = {
            "L1": {
                "both": {"stdev/cov": 1.0, "stdev_type": "stdev"}
            }
        }

        with self.assertRaises(AssertionError):
            validate_processed_property_dict(prop)

        prop = {
            "L1": {
                "both": {"mean": 5, "stdev/cov": 1.0, "stdev_type": "variance"}
            }
        }

        with self.assertRaises(AssertionError):
            validate_processed_property_dict(prop)

        prop = {
            "L1": {
                "wet": {"mean": 5, "stdev/cov": 1.0, "stdev_type": "stdev"}
                # missing dry
            }
        }

        with self.assertRaises(AssertionError):
            validate_processed_property_dict(prop)

        with self.assertRaises(AssertionError):
            validate_processed_property_dict(["not", "a", "dict"])
    
if __name__ == "__main__":
    unittest.main()
