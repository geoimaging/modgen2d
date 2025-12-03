# This file is part of <PROJECT> a Python package for <DESCRIPTION>
# Copyright (C) XXXX Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
# LICENSE

"Create basic sanity checks for project."

import geomodgen2d
import numpy as np
import unittest

class TestBoundaryCreator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 2D domain with default units
        cls.domain2D1 = geomodgen2d.discretized_domain2d.DiscretizedDomain2D(
            span_x=5, span_z=4, dx=1, dz=1
        )
        
        cls.boundary2D1 = geomodgen2d.interfaces_creator2d.AbstractInterfacesCreator2D(
            domain=cls.domain2D1, n_interfaces=3, rng=np.random.default_rng(2))
        
    
        cls.boundary2D = geomodgen2d.interfaces_creator2d.AbstractInterfacesCreator2D(
            domain=cls.domain2D1, n_interfaces=0, rng=np.random.default_rng(2))
                
        cls.boundary_sett= {
    'generator_settings_dict':{
                 'generator_option':'uniform',    # options: 'uniform', 'normal', 'fbm'
                 'z_max_change_per_m':4.5,   # Required for 'uniform' only
                 'H':0.75,  # Required for 'fbm' only
                 'method':'daviesharte',   # Required for 'fbm' only
                 'length':15,   # Required for 'fbm' only
                 'std':2     # Required for 'normal' only Mean has to be zero
                },
    'random_init_boundary_option':'random_sort', #options: 'random_sum', 'random_sort', 'equidistant'
    'filter_settings_dict': {
                 'filter_window_length':3, 
                 'filter_polyorder':2,
                        },
    'boundary_overlap_bottom_priority':True,
    }
           
    def test_rough_boundary(self):
        # Note: no runtime error checks for these random generators
        random_generator_settings_dict = {
                 'generator_option':'uniform',    # options: 'uniform', 'normal', 'fbm'
                 'max_dz_per_unit_length':1.5,}
        self.boundary2D1.generate_rough_interfaces(random_generator_settings_dict)
        assert self.boundary2D1.io.interfaces_matrix.shape == (5,3)

        random_generator_settings_dict = {
                 'generator_option':'normal',    # options: 'uniform', 'normal', 'fbm'
                 'std':0.2,}
        self.boundary2D1.generate_rough_interfaces(random_generator_settings_dict)

        random_generator_settings_dict = {
                 'generator_option':'fbm',    # options: 'uniform', 'normal', 'fbm'
                 'H':0.75,  # Required for 'fbm' only
                 'method':'daviesharte',   # Required for 'fbm' only
                 'length':15,   # Required for 'fbm' only
        }
        self.boundary2D1.generate_rough_interfaces(random_generator_settings_dict)
        
    def test_edit_boundary(self):
        # When Correct
        np.testing.assert_array_equal(self.boundary1D.boundary_array, np.array([[0.], [0.]]))
        np.testing.assert_array_equal(self.boundary2D.boundary_array, np.zeros((self.boundary2D.n_layers-1, len(self.boundary2D.x_ranges))))
        self.boundary1D.edit_boundary_matrix(np.array([[1],[2]]))
        np.testing.assert_array_equal(self.boundary1D.boundary_array, np.array([[1.], [2.]]))
        
        # Assertion Error when incorrect array size.
        self.assertRaises(AssertionError, self.boundary1D.edit_boundary_matrix, np.array([[1,2,3,4,5]]))#, msg="New array must have same shape as previous array")
        
    def test_1D_boundary_creator(self):
        self.boundary1D.edit_boundary_matrix(np.array([[1],[2]]))
        np.testing.assert_array_equal(np.array([[1.],[2.]]), self.boundary1D.boundary_array)
        
        original_boundary = self.boundary1D.boundary_array
        # Assert no changes on filtering and processing boundaries for 1D
        self.boundary1D.generating_boundary(self.boundary_sett['generator_settings_dict'])
        np.testing.assert_array_equal(original_boundary, self.boundary1D.boundary_array)                                      
        self.boundary1D.filtering_boundary(**self.boundary_sett['filter_settings_dict'])
        np.testing.assert_array_equal(original_boundary, self.boundary1D.boundary_array)                                      
        self.boundary1D.processing_boundary(self.boundary_sett['boundary_overlap_bottom_priority'])
        np.testing.assert_array_equal(original_boundary, self.boundary1D.boundary_array)                                      
        self.boundary1D.remeshing_boundary(1.4)
        np.testing.assert_array_equal(original_boundary, self.boundary1D.boundary_array)                                      
        
        # No errors check
        self.boundary1D.boundary_init_points(init_boundary='random_sort') 
        self.boundary1D.boundary_init_points(init_boundary='random_sum') 
        self.boundary1D.boundary_init_points(init_boundary='equidistant') 
        np.testing.assert_array_equal(np.array([[2.],[4.]]), self.boundary1D.boundary_array)
        
        self.boundary1D.boundary_init_points(init_boundary=np.array([2.,3.])) 
        np.testing.assert_array_equal(np.array([[2.],[3.]]), self.boundary1D.boundary_array)    

    def test_2D_boundary_creator(self):
        ##DOING
        # Note: no runtime error checks for these random generators
        self.boundary_sett['generator_settings_dict']['generator_option'] = 'normal'
        self.boundary2D.generating_boundary(self.boundary_sett['generator_settings_dict'])

        self.boundary_sett['generator_settings_dict']['generator_option'] = 'fbm'
        self.boundary2D.generating_boundary(self.boundary_sett['generator_settings_dict'])

        self.boundary_sett['generator_settings_dict']['generator_option'] = 'uniform'
        self.boundary2D.generating_boundary(self.boundary_sett['generator_settings_dict'])
        
        #Checking no runtime errors here
        self.boundary2D.generating_boundary(self.boundary_sett['generator_settings_dict'])
        self.boundary2D.filtering_boundary(**self.boundary_sett['filter_settings_dict'])
        self.boundary2D.processing_boundary(self.boundary_sett['boundary_overlap_bottom_priority'])
        
        original_boundary = self.boundary2D.boundary_array
        self.boundary2D.filtering_boundary(filter_window_length=0, filter_polyorder="NoMatter")
        np.testing.assert_array_equal(original_boundary, self.boundary2D.boundary_array)

        manual_boundary = np.array([[0, 1, 2, 2],
                                    [0, 2, 2, 3],
                                    [0, 1, 3, 4]])
        
        # remeshed_manual_boundary1 = np.array([[0, 0.3333, 0.6667, 1, 1.3333, 1.6667, 2, 2, 2, 2, 1.6667, 1.3333, 1]])
        remeshed_manual_boundary2 = np.array([[0.166667, 1.5     , 2.      ],
                                              [0.333333, 2.      , 2.833333],
                                              [0.166667, 2.      , 3.833333]])
            
        self.boundary2D.edit_boundary_matrix(manual_boundary)
        np.testing.assert_array_equal(manual_boundary, self.boundary2D.boundary_array)
        self.boundary2D.remeshing_boundary(2)
        np.testing.assert_array_almost_equal(remeshed_manual_boundary2, self.boundary2D.boundary_array)                                      
        np.testing.assert_array_equal(self.boundary2D.x_ranges, np.array([1,3,5]))                                      
        np.testing.assert_array_equal(self.boundary2D.del_x, 2)                                      
        # No errors check
        
        self.boundary2D.boundary_init_points(init_boundary=np.array([1, 2.,3.])) 
        remeshed_manual_boundary2+=np.array([[1-1/6],[2-1/3],[3-1/6]])
        np.testing.assert_array_almost_equal(remeshed_manual_boundary2, self.boundary2D.boundary_array)
        self.boundary2D.boundary_init_points(init_boundary='random_sort') 
        self.boundary2D.boundary_init_points(init_boundary='random_sum') 
        self.boundary2D.boundary_init_points(init_boundary='equidistant') 
        np.testing.assert_array_almost_equal([1,2,3], self.boundary2D.boundary_array[:,0])
        
        self.boundary2D.filtering_boundary(**self.boundary_sett['filter_settings_dict'])
        self.boundary2D.processing_boundary(self.boundary_sett['boundary_overlap_bottom_priority'])

    def test_gen_using_def_process(self):
        # No runtime error checks here
        self.boundary1D.gen_using_def_process(self.boundary_sett)
        self.boundary2D.gen_using_def_process(self.boundary_sett)
        
    def test_SurfaceBoundaryCreator(self):
        pass
        
if __name__ == "__main__":
    unittest.main()
