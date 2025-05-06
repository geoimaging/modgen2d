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
        self.assertFalse(f.is_divisible(1,0)) 
        self.assertFalse(f.is_divisible(3.1,2)) 
        self.assertTrue(f.is_divisible(4.,2)) 
    
    def test_check_integer(self):
        self.assertRaises(ValueError, f.check_integer, 2.01)
        self.assertRaises(ValueError, f.check_integer, -2.01)
        self.assertEqual(f.check_integer(4.), 4)
        self.assertEqual(f.check_integer(-2.), -2)
    
    def test_is_integer_value(self):
        self.assertTrue(f.is_integer_value(2.))
        self.assertTrue(f.is_integer_value(-2))
        self.assertTrue(f.is_integer_value('2'))
        self.assertTrue(f.is_integer_value('0'))
        self.assertFalse(f.is_integer_value(2.1))
        self.assertFalse(f.is_integer_value(-2.000001))
        self.assertFalse(f.is_integer_value('2f'))
        
    def test_remeshing_2D_matrix(self):
        pass
    
        
if __name__ == "__main__":
    unittest.main()
