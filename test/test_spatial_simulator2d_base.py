"""Application-level tests for the shared random-field property workflow."""

import numpy as np

from .testing_tools import unittest, TestCase
import modgen2d as mg2d
from modgen2d.lithological_domain2d import LithologicalDomain2DReadOnly
from modgen2d.spatial_simulator2d import CovarianceDecompositionSimulator, SpatialSimulator2DAbstract, load_simulator_from_config


def make_lithological_domain(include_ignored_cells=False):
    """Create a lightweight, validated lithological-domain object."""
    length_config = mg2d.LengthConfig("m", max_grid_density=1000)
    domain = mg2d.DiscretizedDomain2D(2.0, 1.0, 0.25, 0.25, length_config)
    lithology = LithologicalDomain2DReadOnly(domain, "unit_test_soil")
    expected_ids = ["0", "1", "X"] if include_ignored_cells else ["0", "1"]
    lithology._set_lit_ids_expected(expected_ids)
    lithology.gwt_depth = 0.75

    # Two horizontal layers. Assignment through the public property runs the
    # class's ID and domain-shape validation.
    layer_matrix = np.full(domain.shape, "0", dtype="<U1")
    layer_matrix[:, domain.z_centers >= 0.5] = "1"
    if include_ignored_cells:
        layer_matrix[-1, :] = "X"
    lithology.lithological_matrix = layer_matrix
    return lithology

def property_dictionary(lithological_domain, mean, stdev, slope=0.0):
    """Assign one property definition to every layer present in the domain."""
    layer_ids = set(np.unique(lithological_domain.lithological_matrix))
    return {
        layer_id: {
            "both": {
                "mean": mean,
                "mean_slope_with_depth": slope,
                "stdev_or_cov": stdev,
                "stdev_type": "stdev",
            }
        }
        for layer_id in layer_ids
    }

class TestSpatialSimulator2DApplicationWorkflow(TestCase):
    """Exercise the same three-stage workflow used by generated properties."""

    @classmethod
    def setUpClass(cls):
        cls.lithological_domain = make_lithological_domain()

    @classmethod
    def tearDownClass(cls):
        mg2d.GlobalSoilInterfaceConfig.reset()

    def make_simulator(self, seed):
        return CovarianceDecompositionSimulator(
            theta_x=1.0,
            theta_z=0.5,
            rng=np.random.default_rng(seed),
        )

    def test_simulate_zvals_lit_profile_from_lithological_domain(self):
        """A real domain produces a reproducible standardized field of its shape."""
        first = self.make_simulator(101).simulate_zvals_lit_profile_from_lithological_domain(
            self.lithological_domain
        )
        second = self.make_simulator(101).simulate_zvals_lit_profile_from_lithological_domain(
            self.lithological_domain
        )

        self.assertEqual(first.shape, self.lithological_domain.lithological_matrix.shape)
        self.assertTrue(np.all(np.isfinite(first)))
        self.assertArrayEqual(first, second)
        self.assertGreater(np.std(first), 0.0)

    def test_simulate_profile_from_zvals_lit_profile_honors_shared_field(self):
        """One z-field must create the expected cross-property dependence."""
        zvals = self.make_simulator(202).simulate_zvals_lit_profile_from_lithological_domain(
            self.lithological_domain
        )
        simulator = self.make_simulator(999)  # This step must not draw new noise.

        property_a = simulator.simulate_profile_from_zvals_lit_profile(
            zvals,
            self.lithological_domain,
            property_dictionary(self.lithological_domain, mean=10.0, stdev=2.0),
        )
        property_b = simulator.simulate_profile_from_zvals_lit_profile(
            zvals,
            self.lithological_domain,
            property_dictionary(self.lithological_domain, mean=100.0, stdev=5.0),
        )

        self.assertArrayAlmostEqual(property_a, 10.0 + 2.0 * zvals)
        self.assertArrayAlmostEqual(property_b, 100.0 + 5.0 * zvals)
        self.assertAlmostEqual(
            np.corrcoef(property_a.ravel(), property_b.ravel())[0, 1],
            1.0,
        )

        standardized_a = (property_a - 10.0) / 2.0
        standardized_b = (property_b - 100.0) / 5.0
        self.assertArrayAlmostEqual(standardized_a, zvals)
        self.assertArrayAlmostEqual(standardized_b, zvals)

    def test_simulate_profile_from_zvals_lit_profile_applies_depth_trend(self):
        zvals = self.make_simulator(303).simulate_zvals_lit_profile_from_lithological_domain(
            self.lithological_domain
        )
        result = self.make_simulator(404).simulate_profile_from_zvals_lit_profile(
            zvals,
            self.lithological_domain,
            property_dictionary(
                self.lithological_domain,
                mean=20.0,
                stdev=3.0,
                slope=4.0,
            ),
        )
        _, depth = np.meshgrid(
            self.lithological_domain.domain.x_centers,
            self.lithological_domain.domain.z_centers,
            indexing="ij",
        )
        self.assertArrayAlmostEqual(result, 20.0 + 4.0 * depth + 3.0 * zvals)

    def test_simulate_profile_from_lithological_domain_matches_two_stage_workflow(self):
        """The convenience method must equal explicit z-field generation and mapping."""
        properties = property_dictionary(
            self.lithological_domain,
            mean=50.0,
            stdev=4.0,
            slope=2.0,
        )
        direct = self.make_simulator(505).simulate_profile_from_lithological_domain(
            self.lithological_domain,
            processed_property_dict=properties,
        )

        staged_simulator = self.make_simulator(505)
        zvals = staged_simulator.simulate_zvals_lit_profile_from_lithological_domain(
            self.lithological_domain
        )
        staged = staged_simulator.simulate_profile_from_zvals_lit_profile(
            zvals,
            self.lithological_domain,
            processed_property_dict=properties,
        )

        self.assertArrayEqual(direct, staged)

    def test_non_spatial_zvals_are_zero_for_every_real_layer(self):
        zvals = self.make_simulator(606).simulate_zvals_lit_profile_from_lithological_domain(
            self.lithological_domain,
            generate_non_spatial_profile=True,
        )
        self.assertArrayEqual(zvals, np.zeros_like(zvals))

    def test_zval_shape_mismatch_is_rejected_for_real_domain(self):
        wrong_shape = np.zeros((
            self.lithological_domain.lithological_matrix.shape[0] + 1,
            self.lithological_domain.lithological_matrix.shape[1],
        ))
        with self.assertRaises(TypeError):
            self.make_simulator(707).simulate_profile_from_zvals_lit_profile(
                wrong_shape,
                self.lithological_domain,
                property_dictionary(self.lithological_domain, 10.0, 2.0),
            )

    def test_ignored_x_cells_keep_the_required_sentinel_value(self):
        """Ignored cells must remain untouched during both workflow stages."""
        lithology = make_lithological_domain(include_ignored_cells=True)
        simulator = self.make_simulator(808)
        zvals = simulator.simulate_zvals_lit_profile_from_lithological_domain(
            lithology,
            ignore_lithological_ids=["X"],
        )
        ignored = lithology.lithological_matrix == "X"
        retained = ~ignored

        self.assertTrue(np.all(zvals[ignored] == -99999))
        self.assertTrue(np.all(np.isfinite(zvals[retained])))

        properties = property_dictionary(lithology, mean=10.0, stdev=2.0)
        properties.pop("X")
        result = simulator.simulate_profile_from_zvals_lit_profile(
            zvals,
            lithology,
            properties,
            ignore_lithological_ids=["X"],
        )
        self.assertTrue(np.all(result[ignored] == -99999))
        self.assertArrayAlmostEqual(result[retained], 10.0 + 2.0 * zvals[retained])

    def test_wet_and_dry_property_definitions_follow_groundwater_table(self):
        """Exercise the separate wet/dry mean, trend, and spread branch."""
        zvals = self.make_simulator(909).simulate_zvals_lit_profile_from_lithological_domain(
            self.lithological_domain
        )
        layer_ids = np.unique(self.lithological_domain.lithological_matrix)
        properties = {
            layer_id: {
                "dry": {
                    "mean": 10.0,
                    "mean_slope_with_depth": 1.0,
                    "stdev_or_cov": 2.0,
                    "stdev_type": "stdev",
                },
                "wet": {
                    "mean": 20.0,
                    "mean_slope_with_depth": 3.0,
                    "stdev_or_cov": 0.10,
                    "stdev_type": "cov",
                },
            }
            for layer_id in layer_ids
        }
        result = self.make_simulator(1001).simulate_profile_from_zvals_lit_profile(
            zvals,
            self.lithological_domain,
            properties,
            gwt_depth=0.75,
        )
        _, depth = np.meshgrid(
            self.lithological_domain.domain.x_centers,
            self.lithological_domain.domain.z_centers,
            indexing="ij",
        )
        wet = depth >= 0.75
        expected = np.empty_like(zvals)
        expected[~wet] = 10.0 + depth[~wet] + 2.0 * zvals[~wet]
        expected[wet] = 20.0 + 3.0 * depth[wet] + 2.0 * zvals[wet]
        self.assertArrayAlmostEqual(result, expected)

    def test_inconsistent_groundwater_depth_is_rejected(self):
        with self.assertRaises(ValueError):
            self.make_simulator(1101).simulate_zvals_lit_profile_from_lithological_domain(
                self.lithological_domain,
                gwt_depth=0.50,
            )

    def test_missing_layer_property_is_rejected(self):
        zvals = self.make_simulator(1201).simulate_zvals_lit_profile_from_lithological_domain(
            self.lithological_domain
        )
        properties = property_dictionary(self.lithological_domain, 10.0, 2.0)
        properties.pop(next(iter(properties)))
        with self.assertRaises(ValueError):
            self.make_simulator(1202).simulate_profile_from_zvals_lit_profile(
                zvals,
                self.lithological_domain,
                properties,
            )

    def test_property_definition_for_ignored_x_is_rejected(self):
        lithology = make_lithological_domain(include_ignored_cells=True)
        zvals = self.make_simulator(1301).simulate_zvals_lit_profile_from_lithological_domain(
            lithology,
            ignore_lithological_ids=["X"],
        )
        properties = property_dictionary(lithology, 10.0, 2.0)
        with self.assertRaises(KeyError):
            self.make_simulator(1302).simulate_profile_from_zvals_lit_profile(
                zvals,
                lithology,
                properties,
                ignore_lithological_ids=["X"],
            )

    def test_disallowed_simulator_rejects_domain_workflows(self):
        simulator = self.make_simulator(1401)
        simulator.allow_simulation = False
        with self.assertRaises(RuntimeError):
            simulator.simulate_zvals_lit_profile_from_lithological_domain(
                self.lithological_domain
            )
        with self.assertRaises(RuntimeError):
            simulator.simulate_profile_from_lithological_domain(
                self.lithological_domain,
                property_dictionary(self.lithological_domain, 10.0, 2.0),
            )

    def test_get_means_am_bm(self):
        a, b = SpatialSimulator2DAbstract.get_means_am_bm(5)
        self.assertEqual(a, 5)
        self.assertEqual(b, 0)

        a, b = SpatialSimulator2DAbstract.get_means_am_bm([2, 3])
        self.assertEqual(a, 2)
        self.assertEqual(b, 3)

    def test_get_means_am_bm_invalid(self):
        with self.assertRaises(ValueError):
            SpatialSimulator2DAbstract.get_means_am_bm([1, 2, 3])

    def test_check_points_valid(self):
        points = np.array([[0, 0], [1, 1], [2, 2], [3, 1], [0,3]])
        pts = SpatialSimulator2DAbstract.check_points(points)
        self.assertArrayEqual(pts, points)

    def test_check_points_invalid_shape(self):
        with self.assertRaises(ValueError):
            SpatialSimulator2DAbstract.check_points([1, 2, 3])
            
        with self.assertRaises(ValueError):
            SpatialSimulator2DAbstract.check_points(np.array([1, 2]))

        with self.assertRaises(ValueError):
            SpatialSimulator2DAbstract.check_points(np.array([[[1, 2]]]))

    def test_check_points_non_numeric(self):
        with self.assertRaises(TypeError):
            SpatialSimulator2DAbstract.check_points([["a", "b"]])
            

    def test_config_round_trip_preserves_rng_state(self):
        points = np.array([[0, 0], [1, 1], [2, 2], [3, 1], [0,3]])
        sim = CovarianceDecompositionSimulator(1., 2., -999,
                                               np.random.default_rng(42))
        restored = load_simulator_from_config(sim.get_config)
        self.assertEqual(restored.params, {'theta_x': 1., 'theta_z': 2.})
        self.assertArrayEqual(sim.simulate(points), restored.simulate(points))

if __name__ == '__main__':
    unittest.main()