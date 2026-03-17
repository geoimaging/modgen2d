# This file is part of <PROJECT> a Python package for <DESCRIPTION>
# Copyright (C) XXXX Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
# LICENSE

"Create basic sanity checks for project."

import modgen2d
from modgen2d import discretized_interfaces2d
from modgen2d.interface.rough_interface_generator import NormalInterfaceGen, UniformInterfaceGen, FBMInterfaceGen

import numpy as np
from testing_tools import unittest, TestCase

class TestBoundaryCreator(TestCase):
    def setUp(self):
        # 2D domain with default units
        self.domain2D1 = modgen2d.discretized_domain2d.DiscretizedDomain2D(
            span_x=5, span_z=4, dx=1, dz=1
        )
        
        self.boundary2D1 = discretized_interfaces2d.DiscretizedInterfaces2D(
            domain=self.domain2D1, n_soil_layers=3, generate_surface=True, rng=np.random.default_rng(2))
        
        self.domain2D2 = modgen2d.discretized_domain2d.DiscretizedDomain2D(
            span_x=10, span_z=8, dx=0.2, dz=0.1
        )
        
        self.boundary2D2 = discretized_interfaces2d.DiscretizedInterfaces2D(
            domain=self.domain2D2, n_soil_layers=4, generate_surface=False, rng=np.random.default_rng(2))
        
        self.domain2D3 = modgen2d.discretized_domain2d.DiscretizedDomain2D(
            span_x=5, span_z=4, dx=1.25, dz=.8
        )
        self.boundary2D3 = discretized_interfaces2d.DiscretizedInterfaces2D(domain=self.domain2D3, n_soil_layers=1, generate_surface=True, rng=np.random.default_rng(2))
           
    ## TODO add test for set and get rough_interface.
    
    
    def test_rough_interface(self):
        # Note: no runtime error checks for these random generators
        for boundary2D in [self.boundary2D1, self.boundary2D2, self.boundary2D3]:
            shape = boundary2D.shape
            rough_interface_generator = UniformInterfaceGen(1.5)
            boundary2D.generate_rough_interfaces(rough_interface_generator)
            self.assertTupleEqual(boundary2D.interfaces_matrix.shape, shape)
            
            results = []
            for i in range(boundary2D.interfaces_matrix.shape[1]):
                dx = boundary2D.domain.dhs[0]
                interface1D = boundary2D.interfaces_matrix[:,i]
                delz_per_unit_length = np.abs(np.diff(interface1D))/dx
                max_z_per_unit_length = np.max(delz_per_unit_length)
                self.assertTrue(max_z_per_unit_length <= rough_interface_generator.generator_params['max_dz_per_unit_length'], f"Too steep: {max_z_per_unit_length}")
                results.extend(delz_per_unit_length)

            # Check that there is variation — i.e., not all identical
            self.assertGreater(np.std(results), 0.0, "No variation detected — looks non-random")
                
            rough_interface_generator = NormalInterfaceGen(0.2)
            boundary2D.generate_rough_interfaces(rough_interface_generator)
            self.assertTupleEqual(boundary2D.interfaces_matrix.shape, shape)
            results = []
            for i in range(boundary2D.interfaces_matrix.shape[1]):
                interface1D = boundary2D.interfaces_matrix[:,i]
                delz_per_unit_length = np.abs(np.diff(interface1D))
                results.extend(delz_per_unit_length)

            # Check that there is variation — i.e., not all identical
            self.assertGreater(np.std(results), 0.0, "No variation detected — looks non-random")
                        
            rough_interface_generator = FBMInterfaceGen(0.75, 15, 'daviesharte')
            boundary2D.generate_rough_interfaces(rough_interface_generator)
            self.assertTupleEqual(boundary2D.interfaces_matrix.shape, shape)
            results = []
            for i in range(boundary2D.interfaces_matrix.shape[1]):
                interface1D = boundary2D.interfaces_matrix[:,i]
                delz_per_unit_length = np.abs(np.diff(interface1D))
                results.extend(delz_per_unit_length)

            # Check that there is variation — i.e., not all identical
            self.assertGreater(np.std(results), 0.0, "No variation detected — looks non-random")
        
    def test_manual_interfaces_matrix(self):
        interfaces_matrix = np.ones((7, 3))
        self.boundary2D1.set_interfaces_matrix(interfaces_matrix)
        self.assertTupleEqual(self.boundary2D1.interfaces_matrix.shape, (7,3))
        
        with self.assertRaises(ValueError):
            self.boundary2D1.set_interfaces_matrix(np.ones((7, 2)))

        with self.assertRaises(ValueError):
            self.boundary2D1.set_interfaces_matrix([[1,2,3],
                                                        [2,1]])
            
        with self.assertRaises(ValueError):
            self.boundary2D1.set_interfaces_matrix([[1,2,3],
                                                    [2,1,1],
                                                    [2,1,1],
                                                    [2,1,1],
                                                    [2,1,1],
                                                    [2,1,1],
                                                    [2,1,np.nan]],
                                                    )
            
        with self.assertRaises(ValueError):
            self.boundary2D2.set_interfaces_matrix(np.ones((52, 3)))

        with self.assertRaises(ValueError):
            self.boundary2D2.set_interfaces_matrix(np.ones((52, 4)))
            
        interfaces_matrix = np.ones((52, 4))
        interfaces_matrix[:,0] = 0
        self.boundary2D2.set_interfaces_matrix(interfaces_matrix)
        self.assertTupleEqual(self.boundary2D2.interfaces_matrix.shape, (52,4))
            
    def test_filtering_interface(self):
        with self.assertRaises(ValueError):
            self.boundary2D1.filtering_interface() # Error for op shape mismatch. i.e send rough interface before filtering

        self.boundary2D1.set_interfaces_matrix(np.array([[2,3,2,1,2,3,4],
                                                        [0,1,4,6,7,2,5],
                                                        [0,1,6,7,5,1,3]]).T)
        self.boundary2D1.filtering_interface(filter_window_length=7)
        self.assertTupleEqual(self.boundary2D1.interfaces_matrix.shape, (7,3))
        
        interfaces_matrix = np.ones(self.boundary2D2.shape)
        interfaces_matrix[:,0] = 0
        self.boundary2D2.set_interfaces_matrix(interfaces_matrix)
        self.boundary2D2.filtering_interface()
        self.assertTupleEqual(self.boundary2D2.interfaces_matrix.shape, self.boundary2D2.shape)

        self.boundary2D3.set_interfaces_matrix(np.ones(self.boundary2D3.shape))
        self.boundary2D3.filtering_interface(3, 2)
        self.assertTupleEqual(self.boundary2D3.interfaces_matrix.shape, self.boundary2D3.shape)
        
    def test_processing_interface(self):
        ## TODO: Improve if can
        with self.assertRaises(ValueError):
            self.boundary2D1.filtering_interface() # Error for op shape mismatch. i.e send rough interface before filtering

        self.boundary2D1.set_interfaces_matrix(np.array([[1,0.5,0,3,3,3,2.2],
                                                        [1.7,1,2,4,4,4,2],
                                                        [1.5,3,5,2,2,2,2.5]]).T)
        
        self.assertTrue(self.boundary2D1.check_if_overlapping_interfaces())
        
        bottom_erosion = True
        self.boundary2D1.processing_interface(bottom_erosion)
        self.assertTupleEqual(self.boundary2D1.interfaces_matrix.shape, (7,3))
        self.assertArrayAlmostEqual(self.boundary2D1.interfaces_matrix, 
                                    np.array([[1,0.5,0,3,3,3,2.2],
                                            [1.7,1,2,4,4,4,2.2],
                                            [1.7,3,5,4,4,4,2.5]]).T)
        # Check if increasing: No criss cross
        self.assertTrue(self.check_if_increasing_matrix(self.boundary2D1.interfaces_matrix))
        self.assertFalse(self.boundary2D1.check_if_overlapping_interfaces())
        
        self.boundary2D1.set_interfaces_matrix(np.array([[1,0.5,0,3,3,3,2.2],
                                                        [1.7,1,2,4,4,4,2],
                                                        [1.5,3,5,2,2,2,2.5]]).T)
        self.assertTrue(self.boundary2D1.check_if_overlapping_interfaces())
        
        bottom_erosion = False
        self.boundary2D1.processing_interface(bottom_erosion)
        self.assertArrayAlmostEqual(self.boundary2D1.interfaces_matrix, 
                                    np.array([[1,0.5,0,3,3,3,2.2],
                                            [1.5,1,2,3,3,3,2.2],
                                            [1.5,3,5,3,3,3,2.5]]).T)
        self.assertTrue(self.check_if_increasing_matrix(self.boundary2D1.interfaces_matrix))
        self.assertFalse(self.boundary2D1.check_if_overlapping_interfaces())
        
        rough_interface_generator = UniformInterfaceGen(1)
        self.boundary2D2.generate_rough_interfaces(rough_interface_generator)
        bottom_erosion = False
        self.boundary2D2.processing_interface(bottom_erosion)
        self.assertTrue(self.check_if_increasing_matrix(self.boundary2D2.interfaces_matrix))
        self.assertFalse(self.boundary2D2.check_if_overlapping_interfaces())
        
        self.boundary2D2.generate_rough_interfaces(rough_interface_generator)
        bottom_erosion = True
        self.boundary2D2.processing_interface(bottom_erosion)
        self.assertTrue(self.check_if_increasing_matrix(self.boundary2D2.interfaces_matrix))
        self.assertFalse(self.boundary2D2.check_if_overlapping_interfaces())
        
    @staticmethod
    def check_if_increasing_matrix(nparray:np.ndarray, one_dim=False):
        if one_dim and nparray.ndim != 1:
            return False
        return np.sum(np.diff(nparray)<0) == 0

    def test_shape(self):
        self.assertTupleEqual(self.boundary2D1.shape, (7,3))
        self.assertTupleEqual(self.boundary2D2.shape, (52,4))
        
    def test_get_reference_points(self):
        for boundary2D in [self.boundary2D1, self.boundary2D2, self.boundary2D3]:
            ref_points = boundary2D.get_reference_points_zs('random')
            self.assertEqual(len(ref_points), boundary2D.n_soil_layers)
            if boundary2D.n_soil_layers>1:
                self.assertGreater(np.std(ref_points), 0.0, "No variation detected — looks non-random")
            self.assertTrue(self.check_if_increasing_matrix(ref_points, one_dim=True))
            
            
            ref_points = boundary2D.get_reference_points_zs('equidistant')
            self.assertEqual(len(ref_points), boundary2D.n_soil_layers)
            if boundary2D.n_soil_layers>1:
                self.assertGreater(np.std(ref_points), 0.0, "No variation detected — looks non-random")
            self.assertTrue(self.check_if_increasing_matrix(ref_points, one_dim=True))
            
            # ref_points = boundary2D.get_reference_points_zs(np.arange(boundary2D.n_soil_layers))
            # self.assertEqual(len(ref_points), boundary2D.n_soil_layers)
    
        
        ref_points = self.boundary2D1.get_reference_points_zs('equidistant')
        self.assertArrayAlmostEqual(ref_points, [0, 4/3, 8/3])

        ref_points = self.boundary2D2.get_reference_points_zs('equidistant')
        self.assertArrayAlmostEqual(ref_points, [0,2,4,6])
        
        # Failure cases    
        
    def test_update_interface_points(self):
        ref_points = self.boundary2D1.get_reference_points_zs('equidistant')
        # print(self.boundary2D1.print())
        # with self.assertRaises(ValueError):
        #     self.boundary2D1.update_interfaces_depth(ref_points)
            
        interfaces_matrix = [[1.3,2.3,3.3],
                             [4,5,6],
                             [4,5,6],
                             [4,5,6],
                             [11,12,13],
                             [2,3,4],
                             [3,2,1]]
        self.boundary2D1.set_interfaces_matrix(interfaces_matrix)
        self.boundary2D1.update_interfaces_depth(ref_points)
        self.assertTupleEqual(self.boundary2D1.interfaces_matrix.shape, (7,3))
        expected_result = np.array([[1,2,3],
                                     [3.7,4.7,5.7],
                                     [3.7,4.7,5.7],
                                     [3.7,4.7,5.7],
                                     [10.7,11.7,12.7],
                                     [1.7,2.7,3.7],
                                     [2.7,1.7,0.7]])
        expected_result+=np.array([0-1,4/3-2,8/3-3])+1.3 #equivalent adjust in org. test [1,2,3] -> [0,4/3,8/3].. 1.3 (adjust for surface.)
        
        self.assertArrayAlmostEqual(expected_result,
                                    self.boundary2D1.interfaces_matrix
                                    )
        
        with self.assertWarnsRegex(
            UserWarning,
            r"Requested position \(-2\.000\) out of domain bound\. Hence, setting to closest edge/bound \(-0\.500\)\."
        ):
            self.boundary2D1.update_interfaces_depth([0, 2, 3], -2)
        
        expected_result = np.array([[1,2,3],
                                [3.7,4.7,5.7],
                                [3.7,4.7,5.7],
                                [3.7,4.7,5.7],
                                [10.7,11.7,12.7],
                                [1.7,2.7,3.7],
                                [2.7,1.7,0.7]])
        expected_result+=np.array([-1,0,0])+1.3  #added -1 because in org. test [1,2,3] -> [0,2,3].. 1.3 (adjust for surface.)
        
        self.assertArrayAlmostEqual(expected_result,
                                    self.boundary2D1.interfaces_matrix
                                    )
        
        self.boundary2D1.update_interfaces_depth([0, 3, 7], 3.8)
        self.assertTupleEqual(self.boundary2D1.interfaces_matrix.shape, (7,3))
        expected_result = np.array([[-6  ,-4  ,0],
                                    [-3.3,-1.3,2.7],
                                    [-3.3,-1.3,2.7],
                                    [-3.3,-1.3,2.7],
                                    [3.7 , 5.7,9.7],
                                    [-5.3,-3.3,0.7],
                                    [-4.3,-4.3,-2.3]])
        expected_result+=np.array([-1,0,0])+7+1.3 #Tocheck if 8.3 is the interp value at 3.8 for surface
        
        self.assertArrayAlmostEqual(expected_result,
                                    self.boundary2D1.interfaces_matrix
                                    )

    def test_lock(self):
        for boundary2D in [self.boundary2D1, self.boundary2D2, self.boundary2D3]:
            self.assertRaises(ValueError, boundary2D.lock_interfaces)
            shape = boundary2D.shape
            rough_interface_generator = UniformInterfaceGen(1)
            boundary2D.generate_rough_interfaces(rough_interface_generator)
            self.assertTupleEqual(boundary2D.interfaces_matrix.shape, shape)
        
            ref_points = boundary2D.get_reference_points_zs('equidistant')
            boundary2D.update_interfaces_depth(ref_points)
            boundary2D.processing_interface()
            boundary2D.lock_interfaces()
        
            self.assertRaises(SystemError, boundary2D.set_interfaces_matrix, boundary2D.interfaces_matrix)
    
    ## TODO test get_seperate
    # def test_get_seperate_interfaces_matrix(self):
    #     self.assertRaises(ValueError, self.boundary2D1.seperate_surface_interface)
    #     self.boundary2D1.set_interfaces_matrix(np.array([[1,0.5,0,3,3,3,2.2],
    #                                                     [1.7,1,2,4,4,4,2],
    #                                                     [1.5,3,5,2,2,2,2.5]]).T)
        
    #     self.boundary2D3.set_interfaces_matrix(np.array([[1,0.5,3,3,3,2.2]]).T)
        
    #     self.assertRaises(ValueError, self.boundary2D3.seperate_surface_interface)
    #     A, B = self.boundary2D1.seperate_surface_interface()
    #     self.assertArrayAlmostEqual(B.interfaces_matrix, 
    #                                 np.array([[1,0.5,0,3,3,3,2.2]]).T)
    #     self.assertArrayAlmostEqual(A.interfaces_matrix,
    #                                         np.array([[1.7,1,2,4,4,4,2],
    #                                                     [1.5,3,5,2,2,2,2.5]]).T)
        
        
    #     C, B = A.seperate_surface_interface()
    #     self.assertArrayAlmostEqual(C.interfaces_matrix, 
    #                                 np.array([[1.5,3,5,2,2,2,2.5]]).T)
        
    #     D = C.get_interfaces_matrix_with_surface(B, "erode")
    #     self.assertArrayAlmostEqual(D.interfaces_matrix,
    #                                 np.array([[0.7,0,1,3,3,3,1],
    #                                           [0.7,2,4,3,3,3,1.5]]).T)
        
    #     A, B = C.seperate_surface_interface()
    #     self.assertArrayAlmostEqual(A.interfaces_matrix, 
    #                                 np.empty((7,0)))      
        
    #     D = A.get_interfaces_matrix_with_surface(B, "erode")
    #     self.assertArrayAlmostEqual(D.interfaces_matrix,
    #                                 C.interfaces_matrix - 1.5)
    
         
    def test_remeshing(self):
        self.domain2D1 = self.domain2D1.remesh(2.5)
        self.boundary2D1 = discretized_interfaces2d.DiscretizedInterfaces2D(
            domain=self.domain2D1, generate_surface=True, n_soil_layers=3, rng=np.random.default_rng(2))
        self.boundary2D1.set_interfaces_matrix([[0.2, 4.6, 2.9],
                                                [1.3, 3.6, 3.6],
                                                [2.4, 2.6, 4.3],
                                                [3.5, 1.6, 5.0],
                                                ])
        remeshed2D1 = self.boundary2D1.remesh_interface(1, 1)
        remeshed_boundary = [[0.53, 4.3, 3.11],
                             [0.97, 3.9, 3.39],
                             [1.41, 3.5, 3.67],
                             [1.85, 3.1, 3.95],
                             [2.29, 2.7, 4.23],
                             [2.73, 2.3, 4.51],
                             [3.17, 1.9, 4.79],]
        
        self.assertArrayAlmostEqual(remeshed2D1.interfaces_matrix, remeshed_boundary)
        self.domain2D1 = self.domain2D1.remesh(1,1)
        self.assertEqual(self.domain2D1, remeshed2D1.domain)        
        
if __name__ == "__main__":
    unittest.main()
