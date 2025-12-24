import geomodgen2d, warnings
import numpy as np
from testing_tools import unittest, TestCase
from geomodgen2d import discretized_interfaces2d

class TestLithologicalDomain2D(TestCase):
    def setUp(self):
        self.domain2D = geomodgen2d.discretized_domain2d.DiscretizedDomain2D(
            span_x=5, span_z=4, dx=1, dz=1
        )
        self.boundary2D = discretized_interfaces2d.DiscretizedInterfaces2D(
            domain=self.domain2D, n_interfaces=3, rng=np.random.default_rng(2))
        
        boundary = [[0.97, 3.39, 3.39],
                    [0.97, 3.39, 3.39],
                        [1.41, 3.5, 3.67],
                        [1.85, 3.1, 3.95],
                        [2.29, 2.7, 4.23],
                        [2.73, 2.73, 4.51],
                        [2.73, 2.73, 4.51]]
        
        self.boundary2D.set_interfaces_matrix(boundary)

        self.domain2D = self.domain2D.remesh(1.25)
        self.surface_boundary2D = discretized_interfaces2d.SurfaceInterface2D(
            domain=self.domain2D, rng=np.random.default_rng(2))
        self.surface_boundary2D.set_interfaces_matrix([[0.6], [0.2], [0], [1], [1.2], [0.5]])
        
        self.obs2D1 = geomodgen2d.obstruction2d.Obstruction2D(dl=0.5, ref_xz_symbolic=('C', 0), snap_to_dl=False)  #Should be same as above
        self.obs2D1.circle_2d(2, 1)

        self.obs2D2 = geomodgen2d.obstruction2d.Obstruction2D(dl=0.5, ref_xz_symbolic=('0', 0), snap_to_dl=False)  #Should be same as above
        self.obs2D2.rectangle_2d(3, 2, 2)


    def test_GSIConfig_set_soil_interface(self):
        # GeneralConfigTest
        glob_config = geomodgen2d.lithological_domain2d.GlobalSoilInterfaceConfig
        glob_config.reset()
        
        # Trying to set a boundary with multiple interfaces as Surface Boundary
        self.assertRaises(ValueError, glob_config.set_soil_interface, self.boundary2D, self.boundary2D)
        self.assertRaises(TypeError, glob_config.set_soil_interface, None, self.boundary2D)
        self.assertRaises(TypeError, glob_config.set_soil_interface, self.domain2D, self.boundary2D)
        self.assertRaises(TypeError, glob_config.set_soil_interface, self.surface_boundary2D, self.boundary2D)
        self.assertRaises(ValueError, glob_config.set_soil_interface, self.boundary2D, self.boundary2D)
        
        glob_config.set_soil_interface(self.boundary2D, self.surface_boundary2D)
        # Trying to attempt to re setting the surface Config without "force set"
        self.assertRaises(RuntimeError, glob_config.set_soil_interface, self.boundary2D, None)
        
        rev_id_init = glob_config.get_revision_id()
        glob_config.set_soil_interface(self.boundary2D, self.surface_boundary2D, force_set=True)
        
        # Assert revision id is changed once GSIConfig surface interface changed.
        self.assertNotAlmostEqual(rev_id_init, glob_config.get_revision_id())
        
        # Now Changing with Variable (if we can duplicate and have multiple config)
        glob_config2 = geomodgen2d.lithological_domain2d.GlobalSoilInterfaceConfig
        self.assertAlmostEqual(glob_config.get_revision_id(), geomodgen2d.lithological_domain2d.GlobalSoilInterfaceConfig.get_revision_id())
        self.assertAlmostEqual(glob_config.get_revision_id(), glob_config2.get_revision_id())
        
        self.assertRaises(RuntimeError, glob_config2.set_soil_interface, self.boundary2D, None)
        self.assertRaises(RuntimeError, geomodgen2d.lithological_domain2d.GlobalSoilInterfaceConfig.set_soil_interface, self.boundary2D, None)      
    
        with self.assertRaises(AttributeError):
            domain = geomodgen2d.lithological_domain2d.GlobalSoilInterfaceConfig._merged_interface2d_instance.domain

        with self.assertRaises(AttributeError):
            geomodgen2d.lithological_domain2d.GlobalSoilInterfaceConfig._merged_interface2d_instance = self.boundary2D
        
        with self.assertRaises(AttributeError):
            geomodgen2d.lithological_domain2d.GlobalSoilInterfaceConfig._merged_interface2d_instance.domain = self.boundary2D.domain

    # def test_GSIConfig_evaluate_consistency(self):
        
    #     glob_config = geomodgen2d.lithological_domain2d.GlobalSoilInterfaceConfig #Short form
    #     rev_id_curr = glob_config.get_revision_id()
    #     # Checking initial conditions (i.e No config defined. revision_id = 0).
    #     self.assertEqual(glob_config.evaluate_consistency(0), 0) #Same config as 
    #     self.assertEqual(rev_id_curr, 0)
    #     self.assertEqual(glob_config._status_code, 0)

    #     # Save previous revision id and then update new config (as if new surface defined.)
    #     rev_id_prev = rev_id_curr 
    #     glob_config.set_surface_interface(self.surface_boundary2D)
    #     rev_id_curr = glob_config.get_revision_id()
    #     self.assertEqual(glob_config.evaluate_consistency(rev_id_prev), 1) # Means changed but not so much changed.
    #     self.assertEqual(glob_config._status_code, 1)
 
    #     # Checking meaning of 1
    #     self.assertNotEqual(glob_config.get_unique_code(rev_id_prev), glob_config.get_unique_code(rev_id_curr))
    #     self.assertFalse(glob_config.get_compute_immediately(rev_id_prev))
    #     self.assertFalse(glob_config.get_compute_immediately(rev_id_curr))
        
    #     rev_id_prev = rev_id_curr 
    #     glob_config.set_surface_interface(self.surface_boundary2D, compute_immediately=True, force_set=True)
    #     rev_id_curr = glob_config.get_revision_id()
    #     self.assertEqual(glob_config.evaluate_consistency(rev_id_prev), 2) # Means changed but not so much changed.
    #     self.assertEqual(glob_config._status_code, 2)
        
    #     # Checking meaning of 2
    #     self.assertNotEqual(glob_config.get_unique_code(rev_id_prev), glob_config.get_unique_code(rev_id_curr))
    #     self.assertFalse(glob_config.get_compute_immediately(rev_id_prev))
    #     self.assertTrue(glob_config.get_compute_immediately(rev_id_curr))
        
    #     rev_id_prev = rev_id_curr 
    #     glob_config.set_surface_interface(self.surface_boundary2D, compute_immediately=True, force_set=True)
    #     rev_id_curr = glob_config.get_revision_id()
    #     self.assertEqual(glob_config.evaluate_consistency(rev_id_prev), 99) # Means changed but not so much changed.
    #     self.assertEqual(glob_config._status_code, 99)
        
    #     # Checking meaning of 99
    #     self.assertNotEqual(glob_config.get_unique_code(rev_id_prev), glob_config.get_unique_code(rev_id_curr))
    #     self.assertTrue(glob_config.get_compute_immediately(rev_id_prev))
    #     # self.assertTrue(glob_config.get_compute_immediately(rev_id_curr))  #Does not matter.
        
    #     rev_id_prev = rev_id_curr 
    #     self.assertEqual(glob_config.evaluate_consistency(rev_id_prev), 0) # Means changed but not so much changed.
    #     self.assertEqual(glob_config._status_code, 99)  ## Though most recent consistency check was flagged 0, but status code is 3, as that is the worst case ever.
    #     # Checking meaning of 0
    #     self.assertEqual(glob_config.get_unique_code(rev_id_prev), glob_config.get_unique_code(rev_id_curr))
    #     self.assertEqual(glob_config.get_compute_immediately(rev_id_prev), glob_config.get_compute_immediately(rev_id_curr))

    def test_lit_domain_interface(self):
        geomodgen2d.lithological_domain2d.GlobalSoilInterfaceConfig.reset()
        geomodgen2d.lithological_domain2d.GlobalSoilInterfaceConfig.set_soil_interface(self.boundary2D, None, 'pile')
        domain = geomodgen2d.lithological_domain2d.GlobalSoilInterfaceConfig.get_interface_instance().domain
        lit = geomodgen2d.lithological_domain2d.LithologicalDomain2D(domain, 1.8)
        lit_matrix = [['1', '2', '2', '4'],
                        ['1', '2', '2', '2'],
                        ['1', '1', '2', '3'],
                        ['1', '1', '2', '3'],
                        ['1', '1', '1', '3']]

        self.assertArrayEqual(lit.lithological_matrix, lit_matrix)
        self.assertDictEqual(lit.get_feature_id_and_lit_val_from_lithological_matrix(), {'def':[1, 2, 3, 4]})
        
        geomodgen2d.lithological_domain2d.GlobalSoilInterfaceConfig.set_soil_interface(self.boundary2D, self.surface_boundary2D, 'pile', True)
        lit = geomodgen2d.lithological_domain2d.LithologicalDomain2D(domain, 1.8)
        lit_matrix_w_surface = [['0', '1', '2', '2'],
                                ['1', '2', '2', '2'],
                                ['0', '1', '1', '2'],
                                ['0', '1', '1', '2'],
                                ['1', '1', '1', '3']]
        
        lit_matrix_w_surface = [['1', '2', '2', '2'], 
                                ['1', '2', '2', '2'], 
                                ['1', '1', '2', '2'], 
                                ['0', '1', '1', '2'], 
                                ['0', '1', '1', '1']]
        
        self.assertArrayEqual(lit.lithological_matrix, lit_matrix_w_surface)
        self.assertDictEqual(lit.get_feature_id_and_lit_val_from_lithological_matrix(), {'def':[0, 1, 2]})
                    
    def test_lit_domain_obs2d_and_merging(self):
        for setting in [1,2,3,4,5]:
        
            remesh_interp_method = 'nearest'
            boundary_dh = 1
            surf_type = 'erode'
            obs_dhx = 0.5
            obs_dhz = 0.5
            
            if setting == 1:
                to_npz = r'./test/data/lit_domain_test_coarse_nearest.npz'
            elif setting == 2:
                remesh_interp_method = 'linear'
                surf_type = 'pile'
                to_npz = r'./test/data/lit_domain_test_coarse_linear.npz'
            elif setting == 3:
                remesh_interp_method = 'nearest'
                boundary_dh = 0.25
                surf_type = 'erode'
                obs_dhx = 0.2
                obs_dhz = 0.1
                to_npz = r'./test/data/lit_domain_test_dense_linear.npz'
            elif setting == 4:
                remesh_interp_method = 'linear'
                boundary_dh = 0.25
                surf_type = 'pile'
                obs_dhx = 0.2
                obs_dhz = 0.1
                to_npz = r'./test/data/lit_domain_test_dense_nearest.npz'
            else:
                surf_type = 'pile'
                to_npz = r'./test/data/lit_domain_test_coarse_nearest_pile.npz'

            gwt_depth = 1.8
            
            self.boundary2D.remesh_interp_method = remesh_interp_method
            self.surface_boundary2D.remesh_interp_method = remesh_interp_method
            
            data = np.load(to_npz)
            loaded_dict = {key: data[key] for key in data.files}

            self.assertArrayAlmostEqual(loaded_dict['boundary2D_matrix'], self.boundary2D.interfaces_matrix)
            self.assertArrayAlmostEqual(loaded_dict['surfboundary2D_matrix'], self.surface_boundary2D.interfaces_matrix)

            geomodgen2d.lithological_domain2d.GlobalSoilInterfaceConfig.reset()
            geomodgen2d.lithological_domain2d.GlobalSoilInterfaceConfig.set_soil_interface(self.boundary2D, self.surface_boundary2D, surf_type)
            
            domain2D_lit = geomodgen2d.discretized_domain2d.DiscretizedDomain2D(
                span_x=5, span_z=4, dx=boundary_dh, dz=boundary_dh
            )

            name = 'lit_domain1'
            lit = geomodgen2d.lithological_domain2d.LithologicalDomain2D(domain2D_lit, gwt_depth, name)

            domain2D_obs = geomodgen2d.discretized_domain2d.DiscretizedDomain2D(
                        span_x=5, span_z=4, dx=obs_dhx, dz=obs_dhz
                    )       
            
            obs1_ref_point = [1.25, .74] #edge cases... minus .01 to the left 
            obs2_ref_point = [1.38,1.44]

            obs_lit = geomodgen2d.lithological_domain2d.LithologicalDomain2DFromObstruction2D(domain2D_obs, 'obst')
            # obs_lit.add_obstruction2D(obs2D1, [1.4,1.7], 'U')
            obs_lit.add_obstruction2D(self.obs2D1, obs1_ref_point, 'U')
            obs_lit.add_obstruction2D(self.obs2D2, obs2_ref_point, 'U')

            self.assertArrayEqual(loaded_dict['obs_lit_matrix'], obs_lit.lithological_matrix)
            
            merged_lit = lit.return_merged_lithological_domain([obs_lit])
            self.assertArrayEqual(loaded_dict['merged_lit_matrix'], merged_lit.lithological_matrix)

            ## Method2 of getting merged_lit
            lit = geomodgen2d.lithological_domain2d.LithologicalDomain2D(domain2D_lit, gwt_depth, name)
            obs_lit1 = geomodgen2d.lithological_domain2d.LithologicalDomain2DFromObstruction2D(domain2D_obs, 'obst')
            obs_lit1.add_obstruction2D(self.obs2D1, obs1_ref_point, 'U')

            obs_lit2 = geomodgen2d.lithological_domain2d.LithologicalDomain2DFromObstruction2D(domain2D_obs, 'obst')
            obs_lit2.add_obstruction2D(self.obs2D2, obs2_ref_point, 'U')

            merged_lit = lit.return_merged_lithological_domain([obs_lit1, obs_lit2])
            self.assertArrayEqual(loaded_dict['merged_lit_matrix'], merged_lit.lithological_matrix)
        
    def test_remeshing(self):
        pass
    
    def test_refresh_equality(self):
        pass

        
if __name__ == "__main__":
    unittest.main()
