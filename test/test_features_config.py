from geomodgen2d.features_config import FeaturesConfig
import numpy as np
from testing_tools import unittest, TestCase
from geomodgen2d.random_generators import DiscreteChoice, Constant

class TestFeaturesConfig(TestCase):
    def test_add_feature_constant(self):
        fc = FeaturesConfig()
        fc.add_feature("def", Constant("sand"), "desc")
        self.assertEqual(fc.get_feature_ids(), ["def"])
        self.assertEqual(fc.get_material_types("def"), ["sand"])
        self.assertIsInstance(
            fc.get_material_types_distribution("def"), Constant
        )
        
        fc.add_feature("utils", DiscreteChoice(["metal", "plastic"], [0.4, 0.6]), "desc2")
        self.assertEqual(fc.get_feature_ids(), ["def", "utils"])
        self.assertEqual(fc.get_material_types("utils"), ["metal", "plastic"])
        self.assertIsInstance(
            fc.get_material_types_distribution("utils"), DiscreteChoice
        )
        fc.check()
        
    def test_add_feature_invalid(self):
        fc = FeaturesConfig()
        fc.add_feature("soil", Constant("sand"))
        
        # Duplicate feature_id
        with self.assertRaises(KeyError):
            fc.add_feature("soil", Constant("clay"))

        # Invalid feature_id
        with self.assertRaises(KeyError):
            fc.add_feature("soil1", Constant("sand"))

        # Invalid generator type
        with self.assertRaises(TypeError):
            fc.add_feature("utils", material_type_distribution=123)
    
        # Test layer0 not allowed
        with self.assertRaises(KeyError):
            fc.add_feature("soilB", DiscreteChoice(["clay", "layer0"]))

        # Test duplicate material types in generator
        with self.assertRaises(ValueError):
            fc.add_feature("soilC", DiscreteChoice(["clay", "clay"]))
            
        # Non string materials
        with self.assertRaises(ValueError):
            fc.add_feature("soilD", DiscreteChoice([1, 2]))
            print(fc.get_material_types("soilD"))
            
        # Test description must be string
        with self.assertRaises(ValueError):
            fc.add_feature("soilE", Constant("sand"), feature_description=123)

    def test_remove_feature(self):
        fc = FeaturesConfig()
        fc.add_feature("A", Constant("sand"))
        fc.remove_feature("A")
        self.assertEqual(fc.get_feature_ids(), [])

    def test_remove_feature_not_exist(self):
        fc = FeaturesConfig()
        with self.assertRaises(KeyError):
            fc.remove_feature("fake")

    def test_check_errors(self):
        fc = FeaturesConfig()
        fc.add_feature("A", DiscreteChoice(["sand", "clay"]))

        # Missing 'def'
        with self.assertRaises(KeyError):
            fc.check()
            
        # Corrupt internal stored list artificially
        fc._feature_ids_mapping["A"]["material_type_list"] = ["sand", "silt"]

        with self.assertRaises(KeyError):
            fc.check()

    def test_getters(self):
        fc = FeaturesConfig()
        fc.add_feature("A", Constant("sand"), "desc1")
        
        # Must have def before getting material_type_distribution
        self.assertRaises(KeyError, fc.get_material_types_distribution, "A")

        fc.add_feature("def", DiscreteChoice(["clay"]), "desc2")
        self.assertEqual(set(fc.get_feature_ids()), {"A", "def"})
        self.assertEqual(
            fc.get_feature_descriptions(),
            {"A": "desc1", "def": "desc2"},
        )
        
        self.assertIsInstance(
            fc.get_material_types_distribution("A"), Constant
        )
        
        fc.reset()
        self.assertEqual(fc.get_feature_ids(), [])
        
if __name__ == "__main__":
    unittest.main()
