"""Tests for material-property configuration and sampling."""

import numpy as np

from .testing_tools import unittest, TestCase
from modgen2d.features_config import FeaturesConfig
from modgen2d.lithological_domain2d import LithologicalDomain2DCollection
from modgen2d.main_properties import (
    AuxillaryProperties,
    MainPropertiesConfig,
    check_consistent_with_lit_domain2d_collection,
)
from modgen2d.main_property_each import MainProperty, PropertyDistribution
from modgen2d.random_generators import Constant


def make_features_config():
    features = FeaturesConfig()
    features.add_feature("def", Constant("sand"), "soil")
    features.add_feature("U", Constant("PVC"), "utility")
    return features


def make_main_property(features, name="Vs"):
    main_property = MainProperty(name, features, layer0_flag=False)
    main_property.add_material_property_of_feature(
        "def",
        "sand",
        PropertyDistribution(
            name,
            Constant(200.0),
            Constant(20.0),
            "stdev",
        ),
    )
    main_property.add_material_property_of_feature(
        "U",
        "PVC",
        PropertyDistribution(
            name,
            Constant(1000.0),
            Constant(0.10),
            "cov",
        ),
    )
    return main_property


def make_locked_lithological_collection():
    """Create only the genuine collection state required for property sampling."""
    collection = LithologicalDomain2DCollection(
        valid_feature_ids=["def", "U"]
    )
    collection._all_lit_ids = {
        "def": [1, 2],
        "U": [1],
    }
    collection._locked = True
    collection._unique_code = 12345
    return collection


class TestMainPropertiesConfig(TestCase):
    def setUp(self):
        self.features = make_features_config()

    def test_initial_state_and_accessors(self):
        config = MainPropertiesConfig(self.features)
        self.assertIs(config.features_config, self.features)
        self.assertFalse(config.layer0_flag)
        self.assertEqual(config.get_feature_ids(), ["def", "U"])
        self.assertEqual(config.get_main_properties_names(), [])
        self.assertEqual(config.lit_id2material_dict, {})
        self.assertEqual(config.sampled_properties, {})
        self.assertEqual(config.unique_code, 0)

    def test_constructor_rejects_invalid_features_config(self):
        with self.assertRaises(TypeError):
            MainPropertiesConfig({"def": "sand"})

    def test_add_main_property_locks_and_rejects_duplicate(self):
        config = MainPropertiesConfig(self.features)
        main_property = make_main_property(self.features)
        config.add_main_property(main_property)

        self.assertIn("Vs", config.main_properties)
        self.assertTrue(main_property.locked)
        self.assertEqual(config.get_main_properties_names(), ["Vs"])
        with self.assertRaises(ValueError):
            config.add_main_property(main_property)

    def test_lock_generates_material_mapping_and_sampled_properties(self):
        config = MainPropertiesConfig(self.features)
        config.add_main_property(make_main_property(self.features, "Vs"))
        config.add_main_property(make_main_property(self.features, "rho"))
        lithologies = make_locked_lithological_collection()

        config.lock_and_generate_sample_properties(lithologies)

        self.assertTrue(config._locked)
        self.assertNotEqual(config.unique_code, 0)
        self.assertEqual(config._lit_collection_unique_code, lithologies.unique_code)
        self.assertEqual(set(config.lit_id2material_dict), {"1", "2", "U_1"})
        self.assertArrayEqual(config.lit_id2material_dict["1"], ["def", "sand"])
        self.assertArrayEqual(config.lit_id2material_dict["U_1"], ["U", "PVC"])
        self.assertEqual(set(config.sampled_properties), {"Vs", "rho"})
        self.assertEqual(set(config.sampled_properties["Vs"]), {"1", "2", "U_1"})
        self.assertEqual(config.sampled_properties["Vs"]["1"]["both"]["mean"], 200.0)
        self.assertEqual(config.sampled_properties["Vs"]["U_1"]["both"]["mean"], 1000.0)
        self.assertIsNone(config.check_consistent_with_lit_domain2d_collection(lithologies))

    def test_locked_configuration_cannot_generate_twice(self):
        config = MainPropertiesConfig(self.features)
        config.add_main_property(make_main_property(self.features))
        lithologies = make_locked_lithological_collection()
        config.lock_and_generate_sample_properties(lithologies)
        with self.assertRaises(ValueError):
            config.lock_and_generate_sample_properties(lithologies)

    def test_generation_rejects_invalid_or_unlocked_collection(self):
        config = MainPropertiesConfig(self.features)
        config.add_main_property(make_main_property(self.features))
        with self.assertRaises(TypeError):
            config.lock_and_generate_sample_properties(object())

        unlocked = LithologicalDomain2DCollection(["def", "U"])
        with self.assertRaises(ValueError):
            config.lock_and_generate_sample_properties(unlocked)

    def test_unlock_clears_generated_state(self):
        config = MainPropertiesConfig(self.features)
        config.add_main_property(make_main_property(self.features))
        config.lock_and_generate_sample_properties(
            make_locked_lithological_collection()
        )
        config.unlock()
        self.assertFalse(config._locked)
        self.assertEqual(config.lit_id2material_dict, {})
        self.assertEqual(config.sampled_properties, {})
        self.assertEqual(config.unique_code, 0)
        self.assertEqual(config._lit_collection_unique_code, 0)

    def test_get_sample_property_rejects_duplicate_generation(self):
        config = MainPropertiesConfig(self.features)
        config.add_main_property(make_main_property(self.features))
        config.lock_and_generate_sample_properties(
            make_locked_lithological_collection()
        )
        with self.assertRaises(AssertionError):
            config._get_sample_property("Vs")

    def test_consistency_rejects_unique_code_mismatch(self):
        config = MainPropertiesConfig(self.features)
        config.add_main_property(make_main_property(self.features))
        lithologies = make_locked_lithological_collection()
        config.lock_and_generate_sample_properties(lithologies)
        lithologies._unique_code += 1
        with self.assertRaises(ValueError):
            config.check_consistent_with_lit_domain2d_collection(lithologies)

    def test_consistency_function_rejects_property_and_layer_mismatches(self):
        lithologies = make_locked_lithological_collection()
        material_dict = {"1": np.array(["def", "sand"])}
        valid_property = {
            "Vs": {"1": {"both": {
                "mean": 200.0,
                "mean_slope_with_depth": 0.0,
                "stdev_or_cov": 20.0,
                "stdev_type": "stdev",
            }}}
        }

        with self.assertRaises(ValueError):
            check_consistent_with_lit_domain2d_collection(
                lithologies, material_dict, valid_property, ["Vs", "rho"]
            )

        invalid_layers = {"Vs": {"2": valid_property["Vs"]["1"]}}
        with self.assertRaises(ValueError):
            check_consistent_with_lit_domain2d_collection(
                lithologies, material_dict, invalid_layers, ["Vs"]
            )

        with self.assertRaises(TypeError):
            check_consistent_with_lit_domain2d_collection(
                object(), material_dict, valid_property, ["Vs"]
            )


class TestAuxillaryProperties(TestCase):
    def test_add_and_reject_duplicate_auxiliary_property(self):
        auxiliary = AuxillaryProperties()
        generator = Constant(4)
        auxiliary.add_aux_property("n_layers", generator)
        self.assertIs(auxiliary.aux_properties["n_layers"], generator)
        with self.assertRaises(AssertionError):
            auxiliary.add_aux_property("n_layers", Constant(5))


if __name__ == "__main__":
    unittest.main()