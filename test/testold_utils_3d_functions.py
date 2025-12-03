import geomodgen2d
import numpy as np
from testing_tools import unittest, TestCase
import geomodgen2d.utils_3d_functions as utils_3d_f
import geomodgen2d.obstruction_2d as obstruction_2d

class TestGeneralFunctions(TestCase):
    
    def test_get_rotation_matrix(self):
        self.assertRaises(TypeError, utils_3d_f.get_rotation_matrix, None)
        np.testing.assert_array_equal(utils_3d_f.get_rotation_matrix([0,0,45])[1:,1:], 2**-0.5 * np.array([[1, -1], [1, 1]]))
        np.testing.assert_array_equal(utils_3d_f.get_rotation_matrix([0,0,-45])[1:,1:], 2**-0.5 * np.array([[1, 1], [-1, 1]]))
        A = np.array([[0.433013, 0.75, 0.5],
                      [-0.866025, 0.5, 0],
                      [-0.25, -0.433013, 0.866025]])
        np.testing.assert_array_almost_equal(utils_3d_f.get_rotation_matrix([-60,30,0]), A)
        
    def test_check_rotation_angles(self):
        self.assertFalse(utils_3d_f.check_rotation_angles(None))
        self.assertFalse(utils_3d_f.check_rotation_angles([0,0,0]))
        self.assertRaises(AssertionError, utils_3d_f.check_rotation_angles, [10])
        self.assertRaises(AssertionError, utils_3d_f.check_rotation_angles, [10, 20])
        self.assertRaises(AssertionError, utils_3d_f.check_rotation_angles, [10, 20, 'x'])
        self.assertTrue(utils_3d_f.check_rotation_angles([0,20,0]))
        
    def test_get_direction_cosines_from_ref_points(self):
        l, angles = utils_3d_f.get_direction_cosines_from_ref_points([-5, 4, -2], [3, 2, 1])
        np.testing.assert_array_almost_equal(l, np.array([3, -2, 8])*1/77**0.5)
        
        l, angles = utils_3d_f.get_direction_cosines_from_ref_points([-5, 4, -2], [-5, 4, 1])
        np.testing.assert_array_almost_equal(l, np.array([1, 0, 0]))


    def test_get_y_projection(self):
        y_target = 10.0
        all_z0_y0_x0 = [[0,1,2,3,4],
                        [3,3,4,5,8],
                        [5,5,6,7,8]]
        common_angle = [90,0,90] #Parallel to y-axis
        z,y,x = utils_3d_f.get_y_projection(y_target, all_z0_y0_x0, common_angle)
        np.testing.assert_array_almost_equal(z, np.array([0, 1, 2, 3, 4]))
        np.testing.assert_array_almost_equal(y, np.array([10, 10, 10, 10, 10]))
        np.testing.assert_array_almost_equal(x, np.array([5, 5, 6, 7, 8]))
        
        common_angle = [0,90,90] #Parallel to y-axis
        self.assertRaises(AssertionError, utils_3d_f.get_y_projection, y_target, all_z0_y0_x0, common_angle)
        
        a = np.arccos(1/3**0.5)*180/np.pi
        common_angle = [a,a,a] #Parallel to y-axis
        z,y,x = utils_3d_f.get_y_projection(y_target, all_z0_y0_x0, common_angle)
        np.testing.assert_array_almost_equal(z, np.array([7, 8, 8, 8, 6]))
        np.testing.assert_array_almost_equal(y, np.array([10, 10, 10, 10, 10]))
        np.testing.assert_array_almost_equal(x, np.array([12, 12, 12, 12, 10]))
        
        common_angle = [30,30,30] #Invalid angle
        self.assertRaises(AssertionError, utils_3d_f.get_y_projection, y_target, all_z0_y0_x0, common_angle)
        
        y_target = 0
        common_angle = [45,45,90] #45 degree rotation about z axis
        z,y,x = utils_3d_f.get_y_projection(y_target, all_z0_y0_x0, common_angle)
        np.testing.assert_array_almost_equal(z, np.array([0, 1, 2, 3, 4]))
        np.testing.assert_array_almost_equal(y, np.array([0, 0, 0, 0, 0]))
        np.testing.assert_array_almost_equal(x, np.array([2, 2, 2, 2, 0]))
        
    def test_get_table_from_utils2d(self):
        Utils2D_1 = obstruction_2d.Utils2D(del_x=0.8, del_z=1.6, refining_factor=2)  
        Utils2D_1.rectangle_2d(lx=1.2, lz=1.6, util_id=2)
        np.testing.assert_array_equal(Utils2D_1.grid, np.array([[2,2,2,2],[2,2,2,2],[2,2,2,2]]))
        self.assertListAlmostEqual(Utils2D_1.ref_coord2d_in_grids,[1,2])
        #looks good based on graph

if __name__ == "__main__":
    unittest.main()
