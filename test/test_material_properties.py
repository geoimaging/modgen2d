import geomodgen2d
import numpy as np
from testing_tools import unittest, TestCase
import geomodgen2d.general_functions as f
from geomodgen2d.features_config import FeaturesConfig
from geomodgen2d.random_generators import RandomGeneratorAbstract, DiscreteChoice, Constant
from geomodgen2d.material_properties import PropertyDistribution, MainProperty
class TestPropertyDistribution(TestCase):
    
    def setUp(self):
        self.mean = Constant(100)
        self.std = Constant(10)

        self.dist = PropertyDistribution(
            "Vs", self.mean, self.std
        )
        
        self.fc = FeaturesConfig()
        self.fc.add_feature("def", Constant("sand"), "desc")
        self.fc.add_feature("utils", DiscreteChoice(["metal", "plastic"], [0.4, 0.6]), "desc2")
    
    def test_property_distribution_valid(self):
        mean = Constant(10)
        pd2 = PropertyDistribution("vs", mean)
        self.assertEqual(pd2.property_name, "vs")
        self.assertEqual(pd2.stdev_type, "stdev")
        self.assertIsNone(pd2.stdev_distribution)
        pd2.check_class()
        self.assertTrue(pd2.check)
        
    def test_property_distribution_valid_w_stdev(self):
        mean = Constant(10)
        stdev = Constant(2)
        pd = PropertyDistribution("vs", mean, stdev, "cov", "Shear Wave Velocity")
        
        self.assertEqual(pd.property_name, "vs")
        self.assertEqual(pd.stdev_type, "cov")
        self.assertFalse(pd.check)
        pd.check_class()
        self.assertTrue(pd.check)
        
    def test_property_distribution_invalid_type(self):
        mean = Constant(10)
        with self.assertRaises(TypeError):
            PropertyDistribution("vs", mean_distribution=5)

        with self.assertRaises(TypeError):
            PropertyDistribution("vs", mean, np.array([5]))

        with self.assertRaises(ValueError):
            PropertyDistribution("vs", mean, stdev_type='bad_type')
    
    ## Main_Property Class Tests
    def test_init(self):
        mp = MainProperty("Vs", self.fc, False, 'ShearWaveVelocity')
        self.assertEqual(mp.main_property_name, "Vs")
        self.assertEqual(mp.layer0_flag, False)
        self.assertEqual(mp.description, 'ShearWaveVelocity')
        self.assertEqual(mp.locked, False)

    def test_add_material_property_of_feature(self):
        mp = MainProperty("Vs", self.fc, False, 'ShearWaveVelocity')
        mp.add_material_property_of_feature("def", "sand", self.dist)
        self.assertIn("sand", mp.main_property_dist["def"])
        
    def test_add_material_property_of_feature_invalids(self):
        
        dist2 = PropertyDistribution("rho", Constant(1))
        mp = MainProperty("Vs", self.fc, False, 'ShearWaveVelocity')
        # Wrong material property name (mismatch: 'rho' and 'Vs')
        with self.assertRaises(ValueError):
            mp.add_material_property_of_feature("def", "sand", dist2)

        ## Wrong feature_id name
        with self.assertRaises(KeyError):
            mp.add_material_property_of_feature("utils_", "plastic", self.dist)
            
         ## Wrong material name
        with self.assertRaises(ValueError):
            mp.add_material_property_of_feature("utils", "sand", self.dist)

        ## Failed check_class
        with self.assertRaises(TypeError):
            mp.add_material_property_of_feature("def", "sand", 5)
            
        ## Duplicate_material
        mp.add_material_property_of_feature("def", "sand", self.dist)
        with self.assertRaises(ValueError):
            mp.add_material_property_of_feature("def", "sand", self.dist)
            
        ## No 'layer0' in soil as flag is False
        with self.assertRaises(ValueError):
            mp.add_material_property_of_feature("def", "layer0", self.dist)
            
        ## No 'layer0' in other than 'def'
        with self.assertRaises(ValueError):
            mp.add_material_property_of_feature("utils", "layer0", self.dist)


    def test_lock_and_generate(self):
        mp = MainProperty("Vs", self.fc, True, 'ShearWaveVelocity')
        mp.add_material_property_of_feature("def", "sand", self.dist)

        ## Not defined anything yet for "utils".
        with self.assertRaises(ValueError):
            mp.lock_and_check()

        dist2 = PropertyDistribution('Vs', Constant(20), None, 'cov')
        mp.add_material_property_of_feature("utils", "metal", self.dist, dist2)
        
        ## Not defined anything yet for "plastic".
        with self.assertRaises(ValueError):
            mp.lock_and_check()
        
        mp.add_material_property_of_feature("utils", "plastic", self.dist, dist2)
        
        ## Missing 'layer0' in soil as flag is True
        with self.assertRaises(ValueError):
            mp.lock_and_check()
            
        mp.add_material_property_of_feature("def", "layer0", self.dist, dist2)
        mp.lock_and_check()

        out = mp.generate_sample_dict("def", "sand")
        self.assertIn("both", out)
        self.assertIn("mean", out["both"])
        self.assertNotIn("wet", out)
        self.assertNotIn("dry", out)
        self.assertIn("stdev/cov", out["both"])
        
        out = mp.generate_sample_dict("utils", "metal")
        self.assertNotIn("both", out)
        self.assertIn("wet", out)
        self.assertIn("mean", out["wet"])
        self.assertIn("stdev/cov", out["wet"])
        self.assertIn("dry", out)
        self.assertEqual(out["dry"]["mean"], 20)
        self.assertEqual(out["dry"]["stdev/cov"], 0)
        self.assertEqual(out["dry"]["stdev_type"], "cov")

        
if __name__ == "__main__":
    unittest.main()
