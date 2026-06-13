import numpy as np
from .testing_tools import unittest, TestCase

from modgen2d.discretized_domain2d import DiscretizedDomain2D
from modgen2d.obstruction2d import Obstruction2D
from modgen2d.interface import DiscretizedInterfaces2D, GlobalSoilInterfaceConfig
from modgen2d.lithological_domain2d import (
    LithologicalDomain2D,
    LithologicalDomain2DFromObstruction2D,
)

class TestLithologicalDomain2DFromObstruction2D(TestCase):

    def setUp(self):
        GlobalSoilInterfaceConfig.reset()

        self.domain = DiscretizedDomain2D(span_x=5, span_z=4, dx=1, dz=1)

        interface = DiscretizedInterfaces2D(
            domain=self.domain,
            n_soil_layers=3,
            generate_surface=True,
            rng=np.random.default_rng(2),
        )
        interface.set_interfaces_matrix(np.tile([0.0, 1.5, 3.0], (7, 1)))
        interface.lock_interfaces()

        GlobalSoilInterfaceConfig.set_soil_interface(interface)

        self.obs = Obstruction2D(
            dl=1.0,
            ref_xz_symbolic=["o", "o"],
            snap_to_dl=True,
        )
        self.obs.rectangle_2d(lx=2.0, lz=2.0, obstruction_id=1)

    def tearDown(self):
        GlobalSoilInterfaceConfig.reset()

    def test_init(self):
        obs_lit = LithologicalDomain2DFromObstruction2D(self.domain, "utils")

        self.assertEqual(obs_lit.name, "utils")
        self.assertEqual(obs_lit.lm_type, "NA")
        self.assertIsNone(obs_lit.lithological_matrix)
        self.assertEqual(obs_lit.obstruction2d_dict_list, [])
        self.assertFalse(obs_lit.obstruction_overlap)

    def test_add_obstruction2D(self):
        obs_lit = LithologicalDomain2DFromObstruction2D(self.domain, "utils")
        obs_lit.add_obstruction2D(self.obs, [1.0, 1.0], "U")

        self.assertEqual(obs_lit.lm_type, "from_obs2D")
        self.assertTupleEqual(obs_lit.lithological_matrix.shape, self.domain.shape)
        self.assertIn("U_1", obs_lit.lit_ids_expected)
        self.assertIn("X", obs_lit.lit_ids_expected)
        self.assertIn("U_1", np.unique(obs_lit.lithological_matrix))
        self.assertEqual(len(obs_lit.obstruction2d_dict_list), 1)

    def test_add_obstruction2D_def_feature(self):
        obs_lit = LithologicalDomain2DFromObstruction2D(self.domain, "def_obs")
        obs_lit.add_obstruction2D(self.obs, [1.0, 1.0], "U")

        self.assertEqual(obs_lit.lm_type, "from_obs2D")
        self.assertIn("U_1", obs_lit.lit_ids_expected)
        self.assertIn("X", obs_lit.lit_ids_expected)
        self.assertIn("U_1", np.unique(obs_lit.lithological_matrix))

    def test_add_obstruction2D_invalids(self):
        obs_lit = LithologicalDomain2DFromObstruction2D(self.domain, "utils")

        with self.assertRaises(ValueError):
            obs_lit.add_obstruction2D(self.obs, [1.0, 1.0, 1.0], "U")

        with self.assertRaises(ValueError):
            obs_lit.add_obstruction2D(self.obs, [1.0, 1.0], "U_1")

        with self.assertRaises(ValueError):
            obs_lit.add_obstruction2D(self.obs, [1.0, 1.0], "U1")

        undefined_obs = Obstruction2D(dl=1.0)

        with self.assertRaises(AssertionError):
            obs_lit.add_obstruction2D(undefined_obs, [1.0, 1.0], "U")

    def test_refresh(self):
        obs_lit = LithologicalDomain2DFromObstruction2D(self.domain, "utils")
        obs_lit.add_obstruction2D(self.obs, [1.0, 1.0], "U")

        before = obs_lit.lithological_matrix.copy()
        obs_lit.refresh()

        self.assertArrayEqual(obs_lit.lithological_matrix, before)
        self.assertEqual(len(obs_lit.obstruction2d_dict_list), 1)

    def test_merge_with_interface_lithological_domain(self):
        soil_lit = LithologicalDomain2D(self.domain, 1.8, "soil")

        obs_lit = LithologicalDomain2DFromObstruction2D(self.domain, "utils")
        obs_lit.add_obstruction2D(self.obs, [1.0, 1.0], "U")

        merged = soil_lit.return_merged_lithological_domain([obs_lit])

        self.assertTrue(merged.merged_lit)
        self.assertTupleEqual(merged.lithological_matrix.shape, self.domain.shape)
        self.assertIn("U_1", np.unique(merged.lithological_matrix))
        self.assertIn("U", merged.get_feature_id_and_lit_val())

    def test_merge_invalids(self):
        soil_lit = LithologicalDomain2D(self.domain, 1.8, "soil")

        with self.assertRaises(TypeError):
            soil_lit.return_merged_lithological_domain([soil_lit])

    def test_get_config_and_from_config(self):
        obs_lit = LithologicalDomain2DFromObstruction2D(self.domain, "utils")
        obs_lit.add_obstruction2D(self.obs, [1.0, 1.0], "U")

        config = obs_lit.get_config
        recreated = LithologicalDomain2DFromObstruction2D.from_config(config)

        self.assertEqual(recreated.domain, obs_lit.domain)
        self.assertEqual(recreated.name, obs_lit.name)
        self.assertEqual(recreated.lm_type, obs_lit.lm_type)
        self.assertArrayEqual(recreated.lithological_matrix, obs_lit.lithological_matrix)
        self.assertEqual(recreated.lit_ids_expected, obs_lit.lit_ids_expected)

if __name__ == "__main__":
    unittest.main()
