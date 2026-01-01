import geomodgen2d
import numpy as np
from testing_tools import unittest, TestCase

class TestObstruction2D(TestCase):
    @classmethod
    def setUp(self):
        self.obs2D1 = geomodgen2d.obstruction2d.Obstruction2D(dl=0.5, ref_xz_symbolic=['c', 'C'], snap_to_dl=True)  #Should be same as above
        self.obs2D2 = geomodgen2d.obstruction2d.Obstruction2D(dl=0.45, ref_xz_symbolic=('C', 0), snap_to_dl=False)  #Should be same as above

    def test_obstacles_definition_error(self):
        self.assertRaises(ValueError, geomodgen2d.obstruction2d.Obstruction2D, 0) #del_z<=0
        self.assertRaises(ValueError, geomodgen2d.obstruction2d.Obstruction2D, -0.1) #del_x<=0
        self.assertRaises(ValueError, geomodgen2d.obstruction2d.Obstruction2D, 0.2, "a") #del_x<=0
       
    def test_validate_ref_xy_symbolic(self):
        self.assertRaises(ValueError, geomodgen2d.obstruction2d.Obstruction2D.validate_ref_xz_symbolic_format, ['c', 'c', 0]) # dim issue
        self.assertRaises(ValueError, geomodgen2d.obstruction2d.Obstruction2D.validate_ref_xz_symbolic_format, [['c', 'c']]) # dim issue
        self.assertRaises(ValueError, geomodgen2d.obstruction2d.Obstruction2D.validate_ref_xz_symbolic_format, ['c']) # dimension issue
        self.assertRaises(ValueError, geomodgen2d.obstruction2d.Obstruction2D.validate_ref_xz_symbolic_format, np.array(['c', 'c'])) #Only list or tuple not Array
    
    def test_validate_xz_override(self):
        self.assertRaises(ValueError, geomodgen2d.obstruction2d.Obstruction2D.validate_ref_xz_override, ['c', 0]) # Format issue
        self.assertRaises(ValueError, geomodgen2d.obstruction2d.Obstruction2D.validate_ref_xz_override, [1,2,3]) # dim issue
        self.assertRaises(ValueError, geomodgen2d.obstruction2d.Obstruction2D.validate_ref_xz_override, [[2,3]]) # dim issue
        self.assertRaises(ValueError, geomodgen2d.obstruction2d.Obstruction2D.validate_ref_xz_override, ['2','3']) # dim issue
    
    def test_set_manual_ref_xz(self):
        self.assertIsNone(self.obs2D1.ref_xz_override)
        self.assertArrayAlmostEqual(self.obs2D1.ref_xz_symbolic, 
                            [1, 1])
        
        self.obs2D1.set_manual_ref_xz([0, 0.])
        self.assertIsNone(self.obs2D1.ref_xz_symbolic)
        self.assertArrayAlmostEqual(self.obs2D1.get_ref_xz_in_unit_length(), 
                            np.array([0, 0]))
        
        self.obs2D1.set_manual_ref_xz([2.5, -0.5])
        self.assertArrayAlmostEqual(self.obs2D1.get_ref_xz_in_unit_length(), 
                            np.array([2.5, -0.5]))
        self.assertIsNone(self.obs2D1.ref_xz_symbolic)
        
        self.obs2D1.set_manual_ref_xz(["C", 0], symbolic=True)
        self.assertIsNone(self.obs2D1.ref_xz_override)
        self.assertArrayAlmostEqual(self.obs2D1.ref_xz_symbolic, 
                            [1, 0])
        
    def test_rectangle2d(self):
        self.obs2D1.rectangle_2d(2.3, 3, 1)
        self.assertRaises(AssertionError, self.obs2D1.rectangle_2d, 2, 3, 1) #Error shape already defined
        self.assertRaises(ValueError, self.obs2D2.rectangle_2d, -1, 1, 1) #Negative lx
        self.assertRaises(ValueError, self.obs2D2.rectangle_2d, 1, -1, 1) #Negative ly
        self.assertRaises(ValueError, self.obs2D2.rectangle_2d, 1, 1, 0) #Non-positive integer index
        self.assertRaises(ValueError, self.obs2D2.rectangle_2d, 1, 1, 1.2) #Non-positive integer index
        self.assertRaises(ValueError, self.obs2D2.rectangle_2d, 1, 1, -1) #Non-positive integer index
        
        self.assertArrayAlmostEqual(self.obs2D1.grid2d, np.full((5,6), 1))
        self.assertArrayAlmostEqual(self.obs2D1.get_ref_xz_in_unit_length(), 
                            np.array([1.25, 1.5]))
        
        self.obs2D2.rectangle_2d(2.3, 5, 2)
        self.assertArrayAlmostEqual(self.obs2D2.grid2d, np.full((5,11), 2))
        self.assertArrayAlmostEqual(self.obs2D2.get_ref_xz_in_unit_length(), 
                            np.array([1.15, 0]))
        
        self.obs2D1.clear_utils_properties(dl = 0.4, ref_xz_symbolic=[0, 'c'], snap_to_dl=False)
        self.obs2D1.rectangle_2d(lx=.8, lz=.5, obstruction_id=3)
        self.assertArrayAlmostEqual(self.obs2D1.grid2d, np.full((2,1), 3))
        self.assertArrayAlmostEqual(self.obs2D1.get_ref_xz_in_unit_length(), 
                            np.array([0, 0.25]))
        
        self.obs2D1.clear_utils_properties(dl = 0.4, ref_xz_symbolic=[0, 'c'], snap_to_dl=False)
        self.obs2D1.rectangle_2d(lx=0, lz=.5, obstruction_id=3)
        self.assertArrayAlmostEqual(self.obs2D1.grid2d, [])
        self.assertArrayAlmostEqual(self.obs2D1.get_ref_xz_in_unit_length(), 
                            np.array([0, 0.25]))
        
        self.obs2D1.clear_utils_properties(dl = 0.4, ref_xz_symbolic=[0, 'c'], snap_to_dl=False)
        self.obs2D1.rectangle_2d(lx=.8, lz=0, obstruction_id=3)
        self.assertArrayAlmostEqual(self.obs2D1.grid2d, [])
        self.assertArrayAlmostEqual(self.obs2D1.get_ref_xz_in_unit_length(), 
                            np.array([0, 0]))
        
    def test_circle2d(self):
        self.obs2D1.circle_2d(2.3, 1)
        self.assertRaises(AssertionError, self.obs2D1.circle_2d, 2.3, 1) #Error shape already defined
        self.assertRaises(ValueError, self.obs2D2.circle_2d, -1, 1) #Negative d
        self.assertRaises(ValueError, self.obs2D2.circle_2d, 1, 0) #Non-positive integer index
        self.assertRaises(ValueError, self.obs2D2.circle_2d, 1, 1.2) #Non-positive integer index
        self.assertRaises(ValueError, self.obs2D2.circle_2d, 1, -1) #Non-positive integer index
        self.assertArrayAlmostEqual(self.obs2D1.grid2d,
                                    [[0, 1, 1, 1, 0],
                                     [1, 1, 1, 1, 1],
                                     [1, 1, 1, 1, 1],
                                     [1, 1, 1, 1, 1],
                                     [0, 1, 1, 1, 0]] 
                                    )
        self.assertArrayAlmostEqual(self.obs2D1.get_ref_xz_in_unit_length(), 
                            np.array([1.25, 1.25]))
        
        self.obs2D2.circle_2d(2, 1)
        self.assertArrayAlmostEqual(self.obs2D2.grid2d, 
                                    [[0, 1, 1, 1,],
                                     [1, 1, 1, 1,],
                                     [1, 1, 1, 1,],
                                     [1, 1, 1, 1,]] )
        self.assertArrayAlmostEqual(self.obs2D2.get_ref_xz_in_unit_length(), 
                            np.array([1, 0]))
        
        self.obs2D1.clear_utils_properties(dl = 0.4, ref_xz_symbolic=[0, 'c'], snap_to_dl=False)
        self.obs2D1.circle_2d(d=.8, obstruction_id=3)
        self.assertArrayAlmostEqual(self.obs2D1.grid2d, np.full((2,2), 3))
        self.assertArrayAlmostEqual(self.obs2D1.get_ref_xz_in_unit_length(), 
                            np.array([0, 0.4]))
        
        self.obs2D1.clear_utils_properties(dl = 0.2, ref_xz_symbolic=[0, 'c'], snap_to_dl=False)
        self.obs2D1.circle_2d(d=.8, obstruction_id=3)
        self.assertArrayAlmostEqual(self.obs2D1.grid2d, 
                                    [[0,3,3,0],
                                     [3,3,3,3],
                                     [3,3,3,3],
                                     [0,3,3,0]])
        self.assertArrayAlmostEqual(self.obs2D1.get_ref_xz_in_unit_length(), 
                            np.array([0, 0.4]))
        
        self.obs2D1.clear_utils_properties(dl = 0.4, ref_xz_symbolic=[0, 'c'], snap_to_dl=False)
        self.obs2D1.circle_2d(d=0, obstruction_id=3)
        self.assertArrayAlmostEqual(self.obs2D1.grid2d, [])
        self.assertArrayAlmostEqual(self.obs2D1.get_ref_xz_in_unit_length(), 
                            np.array([0, 0]))
        
    def test_shift_one_axis(self):
        self.assertRaises(AssertionError, self.obs2D1.shift_grid_one_axis, 'x', 0) #shape not defined
        
        self.obs2D1.rectangle_2d(lx=.8, lz=1.2, obstruction_id=1)
        self.assertRaises(ValueError, self.obs2D1.shift_grid_one_axis, 'x', -1) #non-negative number when allow_negative_shift False (default)
        self.assertRaises(ValueError, self.obs2D1.shift_grid_one_axis, 'z', -1) #non-negative number when allow_negative_shift False (default)
        self.assertRaises(AssertionError, self.obs2D1.shift_grid_one_axis, 'a', -1, True) #must be either x or z in axis
        
        self.assertArrayAlmostEqual(self.obs2D1.grid2d, 
                            [[1, 1,],
                             [1, 1]] )
        self.assertArrayAlmostEqual(self.obs2D1.get_ref_xz_in_unit_length(), 
                    np.array([.5, .5]))
        
        self.obs2D1.shift_grid_one_axis('x',1.5)
        self.assertArrayAlmostEqual(self.obs2D1.grid2d, 
                            [[0, 0],
                             [0, 0],
                             [0, 0],
                             [1, 1],
                             [1, 1],])
        self.assertArrayAlmostEqual(self.obs2D1.get_ref_xz_in_unit_length(), 
                    np.array([2, .5]))
        
        self.obs2D1.set_manual_ref_xz([0.6, 0.8])
        self.obs2D1.shift_grid_one_axis('z',1.2) #Auto snapped to 1.0
        self.assertArrayAlmostEqual(self.obs2D1.grid2d, 
                    [[0, 0, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 1, 1],
                     [0, 0, 1, 1],] )
        self.assertArrayAlmostEqual(self.obs2D1.get_ref_xz_in_unit_length(), 
                    np.array([0.6, 1.8]))
    
    def test_scale_shapes(self):
        self.assertRaises(AssertionError, self.obs2D1.scale_shapes, 2) #shape not defined
        self.obs2D1.circle_2d(d = 4, obstruction_id=1)
        self.assertRaises(ValueError, self.obs2D1.scale_shapes, -1) #non-negative scaling
        self.assertRaises(ValueError, self.obs2D1.scale_shapes, 0) #non-negative scaling
        
        self.assertArrayAlmostEqual(self.obs2D1.grid2d, 
                            [[0, 0, 1, 1, 1, 1, 0, 0],
                             [0, 1, 1, 1, 1, 1, 1, 0],
                             [1, 1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1, 1],
                             [0, 1, 1, 1, 1, 1, 1, 0],
                             [0, 0, 1, 1, 1, 1, 0, 0]])  ##Bichar gara
        self.assertArrayAlmostEqual(self.obs2D1.get_ref_xz_in_unit_length(), 
                    np.array([2, 2]))
        
        self.obs2D1.scale_shapes(2.4)
        self.assertArrayAlmostEqual(self.obs2D1.grid2d[0], 
                            [0. ,0. ,0. ,0. ,0. ,1. ,1. ,1. ,1. ,1. ,1. ,1. ,1. ,1. ,0. ,0. ,0. ,0. ,0.])
        self.assertArrayAlmostEqual(self.obs2D1.get_ref_xz_in_unit_length(), 
                    np.array([4.8, 4.8]))
        
        self.obs2D1.set_manual_ref_xz([1,1.4])
        self.obs2D1.scale_shapes(0.8)
        self.assertArrayAlmostEqual(self.obs2D1.grid2d[0], 
                            [0. ,0. ,0. ,0. ,1. ,1. ,1. ,1. ,1. ,1. ,1. ,0. ,0. ,0. ,0.])
        self.assertArrayAlmostEqual(self.obs2D1.get_ref_xz_in_unit_length(), 
                    np.array([0.8, 1.12]))
        
    def test_merge_shapes(self):
        pass
    
    def test_obs_description(self):
        pass    
    
    def test_clear_utils_properties(self):
        pass
    
    def test_query_points_in_obstruction(self):
        self.obs2D2.circle_2d(2, 1)

        ## Test obs2d: query_points_in_obstruction
        query_points = [[4.0,6.8], [4.01, 6.79], #Left Bottom Corner (0 due to nearest) and 1(inside)
                        [5, 5.9], [4.6, 5.7], # Inside (1,1)
                        [4, 5], [1, 9], [10, 11], #Outside (0,0,0)
                        [5.79, 5], [5.79, 6.8], #Right corners (0,1 due to nearest 0.5 handling)
                        [5.79, 5.01], [5.79, 6.79], #(1,1)
                    ]
        expected = [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1]

        self.assertArrayEqual(self.obs2D2.query_points_in_obstruction(query_points, [5,5]), expected)
        
        self.assertRaises(AssertionError, self.obs2D1.query_points_in_obstruction, query_points)
        self.assertRaises(ValueError, self.obs2D2.query_points_in_obstruction, query_points, [0,0,0])
        self.assertRaises(ValueError, self.obs2D2.query_points_in_obstruction, [[1,2,3], [4,5]])
        self.assertRaises(ValueError, self.obs2D2.query_points_in_obstruction, [[1,2,3], [4,5,7]])
        self.assertRaises(TypeError, self.obs2D2.query_points_in_obstruction, query_points, ['0',3])
        self.assertRaises(TypeError, self.obs2D2.query_points_in_obstruction, [['1','2']])
        
if __name__ == "__main__":
    unittest.main()
