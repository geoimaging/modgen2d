# This file is part of <PROJECT> a Python package for <DESCRIPTION>
# Copyright (C) XXXX Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
# LICENSE

"Create basic sanity checks for project."

import geomodgen2d
import numpy as np
from testing_tools import unittest, TestCase
import geomodgen2d.utils_2d

class TestUtils2D(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Utils1D_1 = geomodgen2d.utils_2d.Utils1D(lz=6, del_z=0.4, util_id=1)
        cls.Utils1D_2 = geomodgen2d.utils_2d.Utils2D(del_x=0, del_z=0.4)  #Should be same as above
        
        cls.Utils2D_1 = geomodgen2d.utils_2d.Utils2D(del_x=0.2, del_z=0.4)
        cls.Utils2D_2 = geomodgen2d.utils_2d.Utils2D(del_x=0.4, del_z=0.8)
            # span_x=6, span_z=4, del_x=1.5, del_z=0.5, n_layers=4)
    
    def test_utils_shape_assert_checks(self):
        # self.assertRaises(AssertionError, geomodgen2d.utils_2d.Utils2D(del_x=0, del_z=0)) #del_z<=0
        # self.assertRaises(AssertionError, geomodgen2d.utils_2d.Utils2D(del_x=-0.1, del_z=-0.1)) #del_x<=0
        self.assertRaises(AssertionError, self.Utils1D_1.utils_1d, 2, 1) #Error shape already defined
        self.assertRaises(AssertionError, self.Utils1D_1.circular_2d, 2, 1) #Error shape already defined
        self.assertRaises(AssertionError, self.Utils1D_1.rectangle_2d, 2, 3, 1) #Error shape already defined
        self.assertRaises(AssertionError, self.Utils1D_2.circular_2d, 2, 1) #Dim1 error
        self.assertRaises(AssertionError, self.Utils1D_2.rectangle_2d, 2, 3, 1) #Dim1 error
        self.assertRaises(AssertionError, self.Utils2D_1.utils_1d, 2, 1) #Dim2 error
        
    def test_check_util_id(self):
        self.assertEqual(geomodgen2d.utils_2d.check_util_id(2),2)
        self.assertEqual(geomodgen2d.utils_2d.check_util_id(2.00),2)
        self.assertRaises(ValueError, geomodgen2d.utils_2d.check_util_id, 0) 
        self.assertRaises(ValueError, geomodgen2d.utils_2d.check_util_id, 0.3) 
        self.assertRaises(ValueError, geomodgen2d.utils_2d.check_util_id, -1) 
        
    def test_rectangle2d(self):
        self.Utils2D_2.rectangle_2d(lx=1.2, lz=1.6, util_id=2)
        
    def test_shift_grid_both_axes(self):
        self.assertRaises(AssertionError, self.Utils1D_2.shift_grid_both_axes, 0, 1) #shape not defined
        self.assertRaises(AssertionError, self.Utils2D_2.shift_grid_both_axes, 3, 1) #shape not defined
        
        #1D check
        self.Utils1D_2.utils_1d(lz=2, util_id=2)
        self.assertRaises(AssertionError, self.Utils1D_2.shift_grid_both_axes, 2, 1) #cannot shift in x_axis error
        self.assertRaises(ValueError, self.Utils1D_2.shift_grid_both_axes, 0, -1) #non-negative number
        self.assertRaises(ValueError, self.Utils1D_2.shift_grid_both_axes, 0, 0.4) #must be integer
        
        np.testing.assert_array_equal(self.Utils1D_2.grid, np.array([[2],[2],[2],[2],[2]]))
        self.assertListAlmostEqual(self.Utils1D_2.ref_coord,[0,0])
        
        self.Utils1D_2.shift_grid_both_axes(0,2)
        np.testing.assert_array_equal(self.Utils1D_2.grid, np.array([[0],[0],[2],[2],[2],[2],[2]]))
        self.assertListAlmostEqual(self.Utils1D_2.ref_coord,[2,0])
        
        #2D check
        self.Utils2D_2.rectangle_2d(lx=1.2, lz=1.6, util_id=2)
        self.assertRaises(ValueError, self.Utils2D_2.shift_grid_both_axes, -1, 1) #non-negative number
        self.assertRaises(ValueError, self.Utils2D_2.shift_grid_both_axes, -1, -1) #non-negative number
        self.assertRaises(ValueError, self.Utils2D_2.shift_grid_both_axes, 1.2, 1) #must be integer
        self.assertRaises(ValueError, self.Utils2D_2.shift_grid_both_axes, 1, 0.4) #must be integer
        
        np.testing.assert_array_equal(self.Utils2D_2.grid, np.array([[2,2,2],[2,2,2]]))
        self.assertListAlmostEqual(self.Utils2D_2.ref_coord,[0,0])
        
        self.Utils1D_2.shift_grid_both_axes(3,2)
        np.testing.assert_array_equal(self.Utils2D_2.grid, np.array([[0],[0],[2],[2],[2],[2],[2]]))
        self.assertListAlmostEqual(self.Utils2D_2.ref_coord,[2,0])
        
    def test_gen_using_def_process(self):
        # No runtime error checks here
        self.boundary1D.gen_using_def_process(self.boundary_sett)
        self.boundary2D.gen_using_def_process(self.boundary_sett)
        
    def test_SurfaceBoundaryCreator(self):
        pass
        
if __name__ == "__main__":
    unittest.main()
