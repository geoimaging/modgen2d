import numpy as np
from testing_tools import unittest, TestCase
import geomodgen2d.general_functions as f
from geomodgen2d.material_domain2d import MaterialDomain2D, MaterialDomain2DReadOnly
from geomodgen2d.main_properties import MainPropertiesConfig
from geomodgen2d.lithological_domain2d import LithologicalDomain2D, LithologicalDomain2DFromObstruction2D, GlobalSoilInterfaceConfig
from geomodgen2d.generated_model2d import GeneratedModel2D
from unittest.mock import MagicMock, patch

## Most near to model_collection
class TestMaterialDomain2DReadOnly(unittest.TestCase):

    def setUp(self):
        self.main_props = MagicMock(spec=MainPropertiesConfig)
        self.domain = MaterialDomain2DReadOnly(self.main_props)

    def test_init_valid(self):
        self.assertIsInstance(self.domain, MaterialDomain2DReadOnly)
        self.assertEqual(self.domain.generated_model_set, {})
        self.assertIsNone(self.domain.gwt_depth)

    def test_init_invalid_main_properties(self):
        with self.assertRaises(TypeError):
            MaterialDomain2DReadOnly("not_a_config")

    def test_readonly_properties(self):
        self.assertEqual(self.domain.all_lit_ids, {})
        self.assertEqual(self.domain.sampled_properties, {})
        self.assertEqual(self.domain.lit_id2material_dict, {})


class TestMaterialDomain2D(unittest.TestCase):

    def setUp(self):
        self.main_props = MagicMock(spec=MainPropertiesConfig)
        self.features_config = MagicMock()
        self.features_config.get_feature_ids = ["A", "B"]
        self.main_props.features_config = self.features_config
        self.main_props.main_properties = {}

        self.domain = MaterialDomain2D(self.main_props)

    # -------------------------------
    # Unlock behavior
    # -------------------------------
    def test_unlock_readonly_raises(self):
        self.domain._read_only = True
        with self.assertRaises(ValueError):
            self.domain.unlock()

    # -------------------------------
    # Add lithological domain (interface)
    # -------------------------------
    def test_add_interface_domain_valid(self):
        lith = MagicMock(spec=LithologicalDomain2D)
        lith.lm_type = "from_interface_config"
        lith.interface_config_revision_id = 1
        lith.gwt_depth = 5.0
        lith.merged_lit = False
        lith.get_feature_id_and_lit_val_from_lithological_matrix.return_value = {
            "A": [1, 2]
        }

        with patch.object(GlobalSoilInterfaceConfig, "get_config_status", return_value=True):
            self.domain.add_lithological_domain_from_soil_interface_config(lith)

        self.assertIn(self.domain.interface_set_name, self.domain.generated_model_set)
        self.assertEqual(self.domain.gwt_depth, 5.0)

    def test_add_interface_domain_wrong_type(self):
        with self.assertRaises(TypeError):
            self.domain.add_lithological_domain_from_soil_interface_config("bad")

    # -------------------------------
    # Add lithological domain (obstruction)
    # -------------------------------
    def test_add_obstruction_domain_valid(self):
        lith = MagicMock(spec=LithologicalDomain2DFromObstruction2D)
        lith.merged_lit = False
        lith.get_feature_id_and_lit_val_from_lithological_matrix.return_value = {
            "A": [3]
        }

        with patch.object(GlobalSoilInterfaceConfig, "get_config_status", return_value=True):
            self.domain.add_lithological_domain_from_obstruction2d(
                set_name="obs1",
                lithological_domain_from_obstruction2d_instance=lith,
                lit_order=1
            )

        self.assertIn("obs1", self.domain.generated_model_set)

    def test_add_obstruction_duplicate_set(self):
        lith = MagicMock(spec=LithologicalDomain2DFromObstruction2D)
        lith.merged_lit = False
        lith.get_feature_id_and_lit_val_from_lithological_matrix.return_value = {
            "A": [1]
        }

        with patch.object(GlobalSoilInterfaceConfig, "get_config_status", return_value=True):
            self.domain.add_lithological_domain_from_obstruction2d("obs", lith, 1)

        with self.assertRaises(ValueError):
            self.domain.add_lithological_domain_from_obstruction2d("obs", lith, 2)

    # -------------------------------
    # Delete lithological domain
    # -------------------------------
    def test_delete_obstruction_domain(self):
        lith = MagicMock(spec=LithologicalDomain2DFromObstruction2D)
        lith.merged_lit = False
        lith.get_feature_id_and_lit_val_from_lithological_matrix.return_value = {
            "A": [1]
        }

        with patch.object(GlobalSoilInterfaceConfig, "get_config_status", return_value=True):
            self.domain.add_lithological_domain_from_obstruction2d("obs", lith, 1)

        self.domain.delete_lithological_domain_from_obstruction2d("obs")
        self.assertNotIn("obs", self.domain.generated_model_set)

    # -------------------------------
    # lit_order logic
    # -------------------------------
    def test_get_lit_orders(self):
        gm1 = MagicMock(spec=GeneratedModel2D)
        gm1.lit_order = 0

        gm2 = MagicMock(spec=GeneratedModel2D)
        gm2.lit_order = 2

        self.domain._generated_model_set = {
            "a": gm1,
            "b": gm2
        }

        orders = self.domain.get_lit_orders()
        self.assertEqual(list(orders.values()), [0, 2])

    # -------------------------------
    # Lock domain workflow
    # -------------------------------
    def test_lock_domain_generates_unique_code(self):
        # Minimal stubs to bypass merge logic
        self.domain._all_lit_ids = {"A": [0]}
        self.domain._lit_id2material_dict = {"A_0": ["A", "layer0"]}
        self.domain._generated_model_set = {}

        with patch.object(self.domain, "_MaterialDomain2D__get_merged_set"):
            with patch("numpy.random.randint", return_value=123):
                self.domain.lock_domain()

        self.assertEqual(self.domain.unique_code, 123)

    # -------------------------------
    # Sampling
    # -------------------------------
    def test_get_sample_property_duplicate(self):
        self.domain._sampled_properties = {"Vs": {}}
        with self.assertRaises(AssertionError):
            self.domain._get_sample_property("Vs")
            
        
if __name__ == "__main__":
    unittest.main()
