import geomodgen2d, warnings
import numpy as np
from testing_tools import unittest, TestCase
from geomodgen2d import discretized_interfaces2d

class TestLithologicalDomain2D(TestCase):
    def setUp(self):
        self.domain2D = geomodgen2d.discretized_interfaces2d.DiscretizedDomain2D(
            span_x=5, span_z=4, dx=1, dz=1
        )
        
        self.boundary = [[0.6, 1.57, 3.99, 3.99],
        [0.2, 1.17, 3.59, 3.59],
        [0, 1.41, 3.5, 3.67],
        [0, 1.85, 3.1, 3.95],
        [1, 3.29, 3.7, 5.23],
        [1.2, 3.93, 3.93, 5.71],
        [0.5, 3.23, 3.23, 5.01]]
        
        self.obs2D1 = geomodgen2d.obstruction2d.Obstruction2D(dl=0.5, ref_xz_symbolic=('0', 0), snap_to_dl=False)  #Should be same as above
        self.obs2D1.circle_2d(2, 1)

        self.obs2D2 = geomodgen2d.obstruction2d.Obstruction2D(dl=0.5, ref_xz_symbolic=('c', 0), snap_to_dl=False)  #Should be same as above
        self.obs2D2.rectangle_2d(3, 2, 2)


    def test_GSIConfig_set_soil_interface(self):
        
        boundary2D = discretized_interfaces2d.DiscretizedInterfaces2D(
            domain=self.domain2D, n_soil_layers=4, generate_surface=True, rng=np.random.default_rng(2))
        
        boundary2D.set_interfaces_matrix(self.boundary)
        
        
        # GeneralConfigTest
        glob_config = geomodgen2d.lithological_domain2d.GlobalSoilInterfaceConfig
        glob_config.reset()
        
        # Trying to set a boundary with multiple interfaces as Surface Boundary
        self.assertRaises(TypeError, glob_config.set_soil_interface, None)
        self.assertRaises(TypeError, glob_config.set_soil_interface, self.domain2D)
        
        glob_config.set_soil_interface(boundary2D)
        # Trying to attempt to re setting the surface Config without "force set"
        self.assertRaises(RuntimeError, glob_config.set_soil_interface, boundary2D)
        
        rev_id_init = glob_config.get_revision_id()
        glob_config.set_soil_interface(boundary2D, force_set=True)
        
        # Assert revision id is changed once GSIConfig surface interface changed.
        self.assertNotAlmostEqual(rev_id_init, glob_config.get_revision_id())
        
        # Now Changing with Variable (if we can duplicate and have multiple config)
        glob_config2 = geomodgen2d.lithological_domain2d.GlobalSoilInterfaceConfig
        self.assertAlmostEqual(glob_config.get_revision_id(), geomodgen2d.lithological_domain2d.GlobalSoilInterfaceConfig.get_revision_id())
        self.assertAlmostEqual(glob_config.get_revision_id(), glob_config2.get_revision_id())
        
        self.assertRaises(RuntimeError, glob_config2.set_soil_interface, boundary2D, None)
        self.assertRaises(RuntimeError, geomodgen2d.lithological_domain2d.GlobalSoilInterfaceConfig.set_soil_interface, boundary2D, None)      
    
        with self.assertRaises(AttributeError):
            domain = geomodgen2d.lithological_domain2d.GlobalSoilInterfaceConfig._discretized_interface2d_instance.domain

        with self.assertRaises(AttributeError):
            geomodgen2d.lithological_domain2d.GlobalSoilInterfaceConfig._discretized_interface2d_instance = boundary2D
        
        with self.assertRaises(AttributeError):
            geomodgen2d.lithological_domain2d.GlobalSoilInterfaceConfig._discretized_interface2d_instance.domain = boundary2D.domain

    def test_lit_domain_interface(self):
        boundary2D = discretized_interfaces2d.DiscretizedInterfaces2D(
            domain=self.domain2D, n_soil_layers=4, generate_surface=True, rng=np.random.default_rng(2))
        
        boundary2D.set_interfaces_matrix(self.boundary)
        boundary2D_hz, _ = boundary2D.get_surface_and_subsurface_interfaces(True)
        boundary2D_hz2, _ = boundary2D.get_surface_and_subsurface_interfaces(False)
        
        geomodgen2d.lithological_domain2d.GlobalSoilInterfaceConfig.reset()
        geomodgen2d.lithological_domain2d.GlobalSoilInterfaceConfig.set_soil_interface(boundary2D_hz)
        domain = geomodgen2d.lithological_domain2d.GlobalSoilInterfaceConfig.get_interface_instance().domain
        lit = geomodgen2d.lithological_domain2d.LithologicalDomain2D(domain, 1.8)
        lit_matrix = [['1', '2', '2', '4'],
                        ['1', '2', '2', '2'],
                        ['1', '1', '2', '3'],
                        ['1', '1', '2', '3'],
                        ['1', '1', '1', '3']]

        self.assertArrayEqual(lit.lithological_matrix, lit_matrix)
        self.assertDictEqual(lit.get_feature_id_and_lit_val_from_lithological_matrix(), {'def':[1, 2, 3, 4]})
        
        geomodgen2d.lithological_domain2d.GlobalSoilInterfaceConfig.set_soil_interface(boundary2D, True)
        lit = geomodgen2d.lithological_domain2d.LithologicalDomain2D(domain, 1.8)
        lit_matrix_w_surface = [['1', '2', '2', '2'], 
                                ['1', '2', '2', '2'], 
                                ['1', '1', '2', '3'], 
                                ['0', '1', '1', '2'], 
                                ['0', '1', '1', '1']]
        
        self.assertArrayEqual(lit.lithological_matrix, lit_matrix_w_surface)
        self.assertDictEqual(lit.get_feature_id_and_lit_val_from_lithological_matrix(), {'def':[0, 1, 2, 3]})
        
        geomodgen2d.lithological_domain2d.GlobalSoilInterfaceConfig.set_soil_interface(boundary2D_hz2, True)
        lit = geomodgen2d.lithological_domain2d.LithologicalDomain2D(domain, 1.8)
        
        lit_matrix = [['1', '2', '2', '2'], 
                                ['1', '2', '2', '2'], 
                                ['1', '1', '2', '3'], 
                                ['1', '1', '1', '2'], 
                                ['1', '1', '1', '1']]
        
        self.assertArrayEqual(lit.lithological_matrix, lit_matrix)
        self.assertDictEqual(lit.get_feature_id_and_lit_val_from_lithological_matrix(), {'def':[1, 2, 3]})
                  
    def test_lit_domain_obs2d_and_merging(self):
        for setting in [1,2,3,4,5,6]:
            print(setting)
            remesh_interp_method = 'nearest'
            boundary_dh = 0.25
            obs_dhx = 0.2
            obs_dhz = 0.1
                    
            if setting == 1:
                boundary_type = 1
                to_npz = r'./test/data/lit_domain_test_B1_nearest.npz'
            elif setting == 2:
                boundary_type = 2
                to_npz = r'./test/data/lit_domain_test_B2_nearest.npz'
            elif setting == 3:
                boundary_type = 3
                to_npz = r'./test/data/lit_domain_test_B3_nearest.npz'
            elif setting == 4:
                boundary_type = 1
                remesh_interp_method = 'linear'
                to_npz = r'./test/data/lit_domain_test_B1_linear.npz'
            elif setting == 5:
                boundary_type = 2
                remesh_interp_method = 'linear'
                to_npz = r'./test/data/lit_domain_test_B2_linear.npz'
            elif setting == 6:
                boundary_type = 3
                remesh_interp_method = 'linear'
                to_npz = r'./test/data/lit_domain_test_B3_linear.npz'

            boundary2D = discretized_interfaces2d.DiscretizedInterfaces2D(
            domain=self.domain2D, n_soil_layers=4, generate_surface=True, rng=np.random.default_rng(2), remesh_interp_method=remesh_interp_method)
            
            boundary2D.set_interfaces_matrix(self.boundary)
            boundary2D_hz, _ = boundary2D.get_surface_and_subsurface_interfaces(True)
            boundary2D_hz2, _ = boundary2D.get_surface_and_subsurface_interfaces(False)

            gwt_depth = 1.8
            
            data = np.load(to_npz)
            loaded_dict = {key: data[key] for key in data.files}


            if boundary_type==1:
                boundary = boundary2D
            elif boundary_type==2:
                boundary = boundary2D_hz
            else:
                boundary = boundary2D_hz2

            geomodgen2d.lithological_domain2d.GlobalSoilInterfaceConfig.reset()
            geomodgen2d.lithological_domain2d.GlobalSoilInterfaceConfig.set_soil_interface(boundary)
            
            domain2D_lit = geomodgen2d.discretized_domain2d.DiscretizedDomain2D(
                span_x=5, span_z=4, dx=boundary_dh, dz=boundary_dh
            )

            name = 'lit_domain1'
            lit = geomodgen2d.lithological_domain2d.LithologicalDomain2D(domain2D_lit, gwt_depth, name)

            domain2D_obs = geomodgen2d.discretized_domain2d.DiscretizedDomain2D(
                        span_x=5, span_z=4, dx=obs_dhx, dz=obs_dhz
                    )       
            
            obs1_ref_point = [0.25, .74] #edge cases... minus .01 to the left 
            obs2_ref_point = [3.38,1.44]

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
