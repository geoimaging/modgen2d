import geomodgen2d
import numpy as np
from testing_tools import unittest, TestCase
import geomodgen2d.general_functions as f

class TestGeneralFunctions(TestCase):
    
    def test_is_divisible(self):
        self.assertRaises(ValueError, f.is_divisible, 0, 1) 
        self.assertRaises(ValueError, f.is_divisible, 2, 2.1) 
        self.assertRaises(ValueError, f.is_divisible, -1, 2) 
        self.assertRaises(ValueError, f.is_divisible, 1, -2)
        self.assertRaises(TypeError, f.is_divisible, None, -2)
        self.assertFalse(f.is_divisible(1,0)) 
        self.assertFalse(f.is_divisible(3.1,2)) 
        self.assertTrue(f.is_divisible(4.,2)) 
        self.assertFalse(f.is_divisible(4.000001, 2))
        self.assertTrue(f.is_divisible(4.000001, 2, tol=1e-5))
    
    def test_check_integer(self):
        self.assertRaises(ValueError, f.check_integer, 2.01)
        self.assertRaises(ValueError, f.check_integer, -2.01)
        self.assertRaises(TypeError, f.check_integer, None)
        self.assertRaises(ValueError, f.check_integer, (4.000001))
        self.assertEqual(f.check_integer(4.0000000001), 4)
        self.assertEqual(f.check_integer(4.), 4)
        self.assertEqual(f.check_integer(-2.), -2)
    
    def test_is_integer_value(self):
        self.assertRaises(TypeError, f.is_integer_value, None)
        self.assertTrue(f.is_integer_value(2.))
        self.assertTrue(f.is_integer_value(-2))
        self.assertTrue(f.is_integer_value('2'))
        self.assertTrue(f.is_integer_value('0'))
        self.assertFalse(f.is_integer_value(2.1))
        self.assertFalse(f.is_integer_value(-2.000001))
        self.assertFalse(f.is_integer_value('2f'))
        
    def test_is_close(self):
        self.assertFalse(f.is_close(2.0000001, 2.00004))
        self.assertTrue(f.is_close(2.0000001, 2.00004, 1e-3))
        self.assertFalse(f.is_close(10, 1000))
    
    def test_check_util_id(self):
        self.assertRaises(ValueError, f.check_util_id, '-2.')
        self.assertRaises(ValueError, f.check_util_id, '-2.d')
        self.assertEqual(f.check_util_id(1.), 1)
        self.assertEqual(f.check_util_id('2.0'), 2)
        
    # def test_check_util_id(self):
    #     self.assertEqual(geomodgen2d.utils_2d.check_util_id(2),2)
    #     self.assertEqual(geomodgen2d.utils_2d.check_util_id(2.00),2)
    #     self.assertRaises(ValueError, geomodgen2d.utils_2d.check_util_id, 0) 
    #     self.assertRaises(ValueError, geomodgen2d.utils_2d.check_util_id, 0.3) 
    #     self.assertRaises(ValueError, geomodgen2d.utils_2d.check_util_id, -1) 
        
    def test_is_valid_prefix(self):
        self.assertRaises(TypeError, f.is_valid_feature_id, [12,23])
        self.assertTrue(f.is_valid_feature_id(None))
        self.assertFalse(f.is_valid_feature_id(""))
        self.assertFalse(f.is_valid_feature_id('AQWEQWEQE'))
        self.assertFalse(f.is_valid_feature_id('A8A'))
        self.assertFalse(f.is_valid_feature_id('A_'))
        self.assertFalse(f.is_valid_feature_id('_V'))
        self.assertFalse(f.is_valid_feature_id('1'))
        self.assertTrue(f.is_valid_feature_id('ABA'))

    def coordinate_vars(x_ranges, z_ranges):
        pass
        # del_x, del_z = x_ranges[0]*2, z_ranges[0]*2
        # span_x, span_z = x_ranges[-1]+del_x/2, z_ranges[-1]+del_z/2
        # return span_x, span_z, del_x, del_z    
        
    def test_remeshing_2D_matrix(self):
        pass
    
        
if __name__ == "__main__":
    unittest.main()
