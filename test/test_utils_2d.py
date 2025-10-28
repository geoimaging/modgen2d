import geomodgen2d
import numpy as np
from testing_tools import unittest, TestCase

class TestUtils2D(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Utils1D_1 = geomodgen2d.utils_2d.Utils1D(lz=6, del_z=0.4, util_id=2)
        cls.Utils1D_2 = geomodgen2d.utils_2d.Utils2D(del_x=0, del_z=0.4)  #Should be same as above
        
        cls.Utils2D_1 = geomodgen2d.utils_2d.Utils2D(del_x=0.2, del_z=0.4)
        cls.Utils2D_2 = geomodgen2d.utils_2d.Utils2D(del_x=0.4, del_z=0.8)
            # span_x=6, span_z=4, del_x=1.5, del_z=0.5, n_layers=4)
    
    def test_utils_shape_assert_checks(self):
        self.assertRaises(AssertionError, geomodgen2d.utils_2d.Utils2D, 0, 0) #del_z<=0
        self.assertRaises(AssertionError, geomodgen2d.utils_2d.Utils2D, -0.1, -0.1) #del_x<=0
        
    def test_utils_1d(self):
        self.assertRaises(AssertionError, self.Utils1D_1.utils_1d, 2, 1) #Error shape already defined
        self.assertRaises(AssertionError, self.Utils2D_2.utils_1d, 2, 1) #Dim2 error
        self.assertRaises(AssertionError, self.Utils1D_2.utils_1d, -1, 1) #Negative lz
        self.assertRaises(AssertionError, self.Utils1D_2.utils_1d, 0, 1) #Negative lz
        self.assertRaises(ValueError, self.Utils1D_2.utils_1d, 1, 0) #Non-positive integer index
        self.assertRaises(ValueError, self.Utils1D_2.utils_1d, 1, 1.2) #Non-positive integer index
        self.assertRaises(ValueError, self.Utils1D_2.utils_1d, 1, -1) #Non-positive integer index
        
        self.Utils1D_2.utils_1d(lz=6, util_id=2)
        np.testing.assert_array_equal(self.Utils1D_2.grid, np.array([[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2]]))
        self.assertListAlmostEqual(self.Utils1D_2.ref_coord_in_grids,[0,0])
        
        # Asserting both ways of definition is same
        np.testing.assert_array_equal(self.Utils1D_2.grid, self.Utils1D_1.grid)
        self.assertListAlmostEqual(self.Utils1D_2.ref_coord_in_grids, self.Utils1D_1.ref_coord_in_grids)
                                        
        self.Utils1D_2.clear_utils_properties()
        self.Utils1D_2.rectangle_2d(lx=0, lz=6, util_id=2)
        np.testing.assert_array_equal(self.Utils1D_2.grid, self.Utils1D_1.grid)
        self.assertListAlmostEqual(self.Utils1D_2.ref_coord_in_grids,[8,0])
        
        self.Utils1D_2.clear_utils_properties()
        self.Utils1D_2.utils_1d(lz=1.05, util_id=2)
        np.testing.assert_array_equal(self.Utils1D_2.grid, np.array([[2],[2],[2],[2]]))  #Changed lz as it is not exactly divisible by del_x. (Nearest round off integer)!!
        self.assertListAlmostEqual(self.Utils1D_2.ref_coord_in_grids,[0,0])
        
    def test_rectangle2d(self):
        self.assertRaises(AssertionError, self.Utils1D_1.rectangle_2d, 2, 3, 1) #Error shape already defined
        self.assertRaises(AssertionError, self.Utils1D_2.rectangle_2d, 2, 3, 1) #Dim1 error
        self.assertRaises(AssertionError, self.Utils2D_2.rectangle_2d, -1, 1, 1) #Negative lx
        self.assertRaises(AssertionError, self.Utils2D_2.rectangle_2d, 1, -1, 1) #Negative ly
        self.assertRaises(ValueError, self.Utils2D_2.rectangle_2d, 1, 1, 0) #Non-positive integer index
        self.assertRaises(ValueError, self.Utils2D_2.rectangle_2d, 1, 1, 1.2) #Non-positive integer index
        self.assertRaises(ValueError, self.Utils2D_2.rectangle_2d, 1, 1, -1) #Non-positive integer index
        self.assertRaises(ValueError, self.Utils2D_2.rectangle_2d, 1, 1, 1, 'not mid or top') #Error ref value
        # No checks for input parameters, so be more extra careful
        
        self.Utils2D_2.rectangle_2d(lx=1.2, lz=1.6, util_id=2)
        np.testing.assert_array_equal(self.Utils2D_2.grid, np.array([[2,2,2,2],[2,2,2,2],[2,2,2,2]]))
        self.assertListAlmostEqual(self.Utils2D_2.ref_coord_in_grids,[1,2])
        
        self.Utils2D_2.clear_utils_properties()
        self.Utils2D_2.rectangle_2d(lx=.8, lz=1.5, util_id=2)
        np.testing.assert_array_equal(self.Utils2D_2.grid, np.array([[2,2,2],[2,2,2], [2,2,2]]))  #Changed lz as it is not exactly divisible by del_x. (Nearest round off integer)!!
        self.assertListAlmostEqual(self.Utils2D_2.ref_coord_in_grids,[1,1])  #Mid means 1.5, 1.5 but, now it is 1,1 (integer division)
        
        self.Utils2D_2.clear_utils_properties()
        self.Utils2D_2.rectangle_2d(lx=.7, lz=1.5, util_id=2, ref='top')
        np.testing.assert_array_equal(self.Utils2D_2.grid, np.array([[2,2,2],[2,2,2],[2,2,2]]))  #Changed lz as it is not exactly divisible by del_x. (Nearest round off integer)!!
        self.assertListAlmostEqual(self.Utils2D_2.ref_coord_in_grids,[0,1])  #Mid means 1.5, 1.5 but, now it is 1,1 (integer division)
        
    def test_circle2d(self):
        self.assertRaises(AssertionError, self.Utils1D_1.circular_2d, 2, 1) #Error shape already defined
        self.assertRaises(AssertionError, self.Utils1D_2.circular_2d, 2, 1) #Dim1 error
        self.assertRaises(AssertionError, self.Utils2D_2.circular_2d, 0, 1) #Negative r
        self.assertRaises(AssertionError, self.Utils2D_2.circular_2d, -1, 1) #Negative r
        self.assertRaises(ValueError, self.Utils2D_2.circular_2d, 1, 0) #Non-positive integer index
        self.assertRaises(ValueError, self.Utils2D_2.circular_2d, 1, 1.2) #Non-positive integer index
        self.assertRaises(ValueError, self.Utils2D_2.circular_2d, 1, -1) #Non-positive integer index
        self.assertRaises(ValueError, self.Utils2D_2.circular_2d, 1, 1, 'not mid or top') #Error ref value
        # No checks for input parameters, so be more extra careful
        
        self.Utils2D_1.circular_2d(r = .6, util_id=2)
        np.testing.assert_array_equal(self.Utils2D_1.grid, np.array([[0, 0, 0, 2, 0, 0, 0],
                                                                     [0, 2, 2, 2, 2, 2, 0],
                                                                     [0, 2, 2, 2, 2, 2, 0],
                                                                     [0, 0, 0, 2, 0, 0, 0]]))
        self.assertListAlmostEqual(self.Utils2D_1.ref_coord_in_grids,[2,3])
        
        self.Utils2D_1.clear_utils_properties()
        self.Utils2D_1.circular_2d(r = .45, util_id=2)
        np.testing.assert_array_equal(self.Utils2D_1.grid, np.array([[0,0,0,0,0],
                                                                     [0,2,2,2,2],
                                                                     [0,2,2,2,0]]))  #r is fixed.!!
        self.assertListAlmostEqual(self.Utils2D_1.ref_coord_in_grids,[1,2])  #Mid means .45, .45 but, now it is .6,.4 (integer division)
        
        self.Utils2D_1.clear_utils_properties()
        self.Utils2D_1.circular_2d(r=.45, util_id=2, ref='top')
        np.testing.assert_array_equal(self.Utils2D_1.grid, np.array([[0,0,0,0,0],
                                                                     [0,2,2,2,2],
                                                                     [0,2,2,2,0]]))  #r is fixed.!!
        self.assertListAlmostEqual(self.Utils2D_1.ref_coord_in_grids,[0,2])  #Mid means .45, .45 but, now it is .6,.4 (integer division)
        
    
    def test_shift_grid_both_axes(self):
        self.assertRaises(AssertionError, self.Utils1D_2.shift_grid_both_axes, 0, 1) #shape not defined
        self.assertRaises(AssertionError, self.Utils2D_2.shift_grid_both_axes, 3, 1) #shape not defined
        
        #1D check
        self.Utils1D_2.utils_1d(lz=2, util_id=2)
        self.assertRaises(AssertionError, self.Utils1D_2.shift_grid_both_axes, 2, 1) #cannot shift in x_axis error
        self.assertRaises(ValueError, self.Utils1D_2.shift_grid_both_axes, 0, -1) #non-negative number
        self.assertRaises(ValueError, self.Utils1D_2.shift_grid_both_axes, 0, 0.4) #must be integer

        self.Utils1D_2.shift_grid_both_axes(0,2)
        np.testing.assert_array_equal(self.Utils1D_2.grid, np.array([[0],[0],[2],[2],[2],[2],[2],[2]]))
        self.assertListAlmostEqual(self.Utils1D_2.ref_coord_in_grids,[2,0])
        
        #2D check
        self.Utils2D_2.rectangle_2d(lx=.8, lz=1.2, util_id=2)
        self.assertRaises(ValueError, self.Utils2D_2.shift_grid_both_axes, -1, 1) #non-negative number
        self.assertRaises(ValueError, self.Utils2D_2.shift_grid_both_axes, -1, -1) #non-negative number
        self.assertRaises(ValueError, self.Utils2D_2.shift_grid_both_axes, 1.2, 1) #must be integer
        self.assertRaises(ValueError, self.Utils2D_2.shift_grid_both_axes, 1, 0.4) #must be integer
              
        self.Utils2D_2.shift_grid_both_axes(3,2)
        np.testing.assert_array_equal(self.Utils2D_2.grid, np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,2,2,2],[0,0,0,2,2,2]]))
        self.assertListAlmostEqual(self.Utils2D_2.ref_coord_in_grids,[3,4])
    
    def test_shift_one_axis(self):
        self.assertRaises(AssertionError, self.Utils1D_2.shift_grid_one_axis, 'x', 0) #shape not defined
        self.assertRaises(AssertionError, self.Utils2D_2.shift_grid_one_axis, 'x', 3) #shape not defined
        
        #1D check
        self.Utils1D_2.utils_1d(lz=2, util_id=2)
        self.assertRaises(AssertionError, self.Utils1D_2.shift_grid_one_axis, 'x', 1) #cannot shift in x_axis error
        self.assertRaises(ValueError, self.Utils1D_2.shift_grid_one_axis, 'z', -1) #non-negative number when allow_negative_shift False (default)
        self.assertRaises(ValueError, self.Utils1D_2.shift_grid_one_axis, 'z', 0.4) #must be integer
        self.assertRaises(ValueError, self.Utils1D_2.shift_grid_one_axis, 'z', -0.4, True) #must be integer

        self.Utils1D_2.shift_grid_one_axis('z',2)
        np.testing.assert_array_equal(self.Utils1D_2.grid, np.array([[0],[0],[2],[2],[2],[2],[2],[2]]))
        self.assertListAlmostEqual(self.Utils1D_2.ref_coord_in_grids,[2,0])
        
        self.Utils1D_2.clear_utils_properties()
        self.Utils1D_2.utils_1d(lz=2, util_id=2)
        self.Utils1D_2.shift_grid_one_axis('z',-2, True)
        np.testing.assert_array_equal(self.Utils1D_2.grid, np.array([[2],[2],[2],[2]]))
        self.assertListAlmostEqual(self.Utils1D_2.ref_coord_in_grids,[-2,0])
        
        #2D check
        self.Utils2D_2.rectangle_2d(lx=.8, lz=1.2, util_id=2)
        self.assertRaises(ValueError, self.Utils2D_2.shift_grid_one_axis, 'x', -1) #non-negative number when allow_negative_shift False (default)
        self.assertRaises(ValueError, self.Utils2D_2.shift_grid_one_axis, 'z', -1) #non-negative number when allow_negative_shift False (default)
        self.assertRaises(ValueError, self.Utils2D_2.shift_grid_one_axis, 'x', -1.1, True) #must be integer 
        self.assertRaises(ValueError, self.Utils2D_2.shift_grid_one_axis, 'z', 0.4) #must be integer
              
        self.Utils2D_2.shift_grid_one_axis('x',3)
        self.Utils2D_2.shift_grid_one_axis('z',2)
        np.testing.assert_array_equal(self.Utils2D_2.grid, np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,2,2,2],[0,0,0,2,2,2]]))
        self.assertListAlmostEqual(self.Utils2D_2.ref_coord_in_grids,[3,4])
        
        self.Utils2D_2.clear_utils_properties()
        self.Utils2D_2.rectangle_2d(lx=.8, lz=1.2, util_id=2)
        self.Utils2D_2.shift_grid_one_axis('x',-1, True)
        self.Utils2D_2.shift_grid_one_axis('z',2)
        np.testing.assert_array_equal(self.Utils2D_2.grid, np.array([[0,0],[0,0],[2,2],[2,2]]))
        self.assertListAlmostEqual(self.Utils2D_2.ref_coord_in_grids,[3,0])
    
    def test_scale_shapes(self):
        self.Utils1D_2.utils_1d(lz=1.6, util_id=2)
        np.testing.assert_array_equal(self.Utils1D_2.grid, np.array([[2],[2],[2],[2],[2]]))
        self.Utils1D_2.scale_shapes(1.5)
        np.testing.assert_array_equal(self.Utils1D_2.grid, np.array([[2],[2],[2],[2],[2],[2],[2],[2]]))

        #2D check
        self.Utils2D_1.circular_2d(r = .6, util_id=2)
        self.Utils2D_1.scale_shapes(0.75)
        np.testing.assert_array_equal(self.Utils2D_1.grid, np.array([[0, 0, 0, 0, 0],
                                                                     [0, 2, 2, 2, 2],
                                                                     [0, 2, 2, 2, 2]]))

    def test_merge_shapes(self):
        pass
    
    def test_clear_utils_properties(self):
        pass

        
if __name__ == "__main__":
    unittest.main()
