import numpy as np
from .testing_tools import unittest, TestCase

from modgen2d.discretized_domain2d import DiscretizedDomain2D
from modgen2d.obstruction2d import Obstruction2D
from modgen2d.interface import DiscretizedInterfaces2D, GlobalSoilInterfaceConfig
from modgen2d.lithological_domain2d import (
    LithologicalDomain2D,
    LithologicalDomain2DFromObstruction2D,
    LithologicalDomain2DCollection,
)

class TestLithologicalDomain2DCollection(TestCase):

    def setUp(self):
        GlobalSoilInterfaceConfig.reset()

        self.domain = DiscretizedDomain2D(
            span_x=5,
            span_z=4,
            dx=1,
            dz=1,
        )

        interface = DiscretizedInterfaces2D(
            self.domain,
            n_soil_layers=3,
            generate_surface=True,
            rng=np.random.default_rng(2),
        )
        interface.set_interfaces_matrix(np.tile([0.0, 1.5, 3.0], (7, 1)))
        interface.lock_interfaces()

        GlobalSoilInterfaceConfig.set_soil_interface(interface)

        self.soil_lit = LithologicalDomain2D(self.domain, 1.8, "soil")

        obs = Obstruction2D(dl=1.0, ref_xz_symbolic=["o", "o"])
        obs.rectangle_2d(2.0, 2.0, 1)

        self.obs_lit = LithologicalDomain2DFromObstruction2D(self.domain, "utils")
        self.obs_lit.add_obstruction2D(obs, [1.0, 1.0], "U")

    def tearDown(self):
        GlobalSoilInterfaceConfig.reset()

    def test_init(self):
        collection = LithologicalDomain2DCollection(
            valid_feature_ids=["def", "U"],
            interface_set_name="def",
        )

        self.assertEqual(collection.interface_set_name, "def")
        self.assertEqual(collection.valid_feature_ids, ["def", "U"])
        self.assertEqual(collection.lit_domain_set, {})
        self.assertEqual(collection.unique_code, 0)
        self.assertIsNone(collection.gwt_depth)

    def test_add_lithological_domain_from_soil_interface_config(self):
        collection = LithologicalDomain2DCollection(["def", "U"])

        collection.add_lithological_domain_from_soil_interface_config(self.soil_lit)

        self.assertIn("def", collection.lit_domain_set)
        self.assertEqual(collection.lit_domain_set["def"].lit_order, 0)
        self.assertEqual(collection.gwt_depth, 1.8)

    def test_add_lithological_domain_from_soil_interface_config_invalids(self):
        collection = LithologicalDomain2DCollection(["def", "U"])

        with self.assertRaises(TypeError):
            collection.add_lithological_domain_from_soil_interface_config(self.obs_lit)

    def test_add_lithological_domain_from_obstruction2d(self):
        collection = LithologicalDomain2DCollection(["def", "U"])
        collection.add_lithological_domain_from_soil_interface_config(self.soil_lit)

        collection.add_lithological_domain_from_obstruction2d(
            "utils",
            self.obs_lit,
            lit_order=1,
        )

        self.assertIn("utils", collection.lit_domain_set)
        self.assertEqual(collection.lit_domain_set["utils"].lit_order, 1)

    def test_add_lithological_domain_from_obstruction2d_invalids(self):
        collection = LithologicalDomain2DCollection(["def", "U"])
        collection.add_lithological_domain_from_soil_interface_config(self.soil_lit)

        with self.assertRaises(ValueError):
            collection.add_lithological_domain_from_obstruction2d(
                "def",
                self.obs_lit,
                lit_order=1,
            )

        with self.assertRaises(ValueError):
            collection.add_lithological_domain_from_obstruction2d(
                "utils",
                self.obs_lit,
                lit_order=0,
            )

        with self.assertRaises(TypeError):
            collection.add_lithological_domain_from_obstruction2d(
                "bad",
                self.soil_lit,
                lit_order=1,
            )

    def test_delete_lithological_domain_from_obstruction2d(self):
        collection = LithologicalDomain2DCollection(["def", "U"])
        collection.add_lithological_domain_from_soil_interface_config(self.soil_lit)
        collection.add_lithological_domain_from_obstruction2d(
            "utils",
            self.obs_lit,
            lit_order=1,
        )

        collection.delete_lithological_domain_from_obstruction2d("utils")

        self.assertNotIn("utils", collection.lit_domain_set)

    def test_delete_lithological_domain_invalids(self):
        collection = LithologicalDomain2DCollection(["def", "U"])
        collection.add_lithological_domain_from_soil_interface_config(self.soil_lit)

        with self.assertRaises(ValueError):
            collection.delete_lithological_domain_from_obstruction2d("def")

        with self.assertRaises(ValueError):
            collection.delete_lithological_domain_from_obstruction2d("missing")

    def test_lock_collection(self):
        collection = LithologicalDomain2DCollection(["def", "U"])
        collection.add_lithological_domain_from_soil_interface_config(self.soil_lit)
        collection.add_lithological_domain_from_obstruction2d(
            "utils",
            self.obs_lit,
            lit_order=1,
        )

        collection.lock()

        self.assertTrue(collection._locked)
        self.assertNotEqual(collection.unique_code, 0)
        self.assertIsNotNone(collection.merged_lit_domain)
        self.assertIn("def", collection.all_lit_ids)
        self.assertIn("U", collection.all_lit_ids)
        self.assertIn("U_1", np.unique(collection.merged_lit_domain.lithological_matrix))

        with self.assertRaises(SystemError):
            collection.add_lithological_domain_from_obstruction2d(
                "utils2",
                self.obs_lit,
                lit_order=2,
            )

    def test_get_lit_orders(self):
        collection = LithologicalDomain2DCollection(["def", "U"])
        collection.add_lithological_domain_from_soil_interface_config(self.soil_lit)
        collection.add_lithological_domain_from_obstruction2d(
            "utils",
            self.obs_lit,
            lit_order=5,
        )

        self.assertDictEqual(
            collection.get_lit_orders(),
            {"def": 0, "utils": 5},
        )

        self.assertDictEqual(
            collection.get_lit_orders(return_new_order=True),
            {"def": 0, "utils": 1},
        )

    def test_invalid_feature_id_in_collection(self):
        collection = LithologicalDomain2DCollection(["def"])
        collection.add_lithological_domain_from_soil_interface_config(self.soil_lit)

        with self.assertRaises(ValueError):
            collection.add_lithological_domain_from_obstruction2d(
                "utils",
                self.obs_lit,
                lit_order=1,
            )

    def test_get_config_and_from_config(self):
        collection = LithologicalDomain2DCollection(["def", "U"])
        collection.add_lithological_domain_from_soil_interface_config(self.soil_lit)
        collection.add_lithological_domain_from_obstruction2d(
            "utils",
            self.obs_lit,
            lit_order=1,
        )
        collection.lock()

        config = collection.get_config
        recreated = LithologicalDomain2DCollection.from_config(config)

        self.assertEqual(recreated.interface_set_name, collection.interface_set_name)
        self.assertEqual(recreated.valid_feature_ids, collection.valid_feature_ids)
        self.assertEqual(recreated.all_lit_ids, collection.all_lit_ids)
        self.assertEqual(recreated.gwt_depth, collection.gwt_depth)
        self.assertArrayEqual(
            recreated.merged_lit_domain.lithological_matrix,
            collection.merged_lit_domain.lithological_matrix,
        )

if __name__ == "__main__":
    unittest.main()
