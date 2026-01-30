# This file is part of <PROJECT> a Python package for <DESCRIPTION>
# Copyright (C) XXXX Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
# LICENSE

"Create basic sanity checks for project."

import modgen2d
from modgen2d import discretized_interfaces2d
from modgen2d.rough_interface_creator2d import NormalInterfaceGen, UniformInterfaceGen, FBMInterfaceGen

import numpy as np
from testing_tools import unittest, TestCase

class TestBoundaryCreator(TestCase):
    def setUp(self):
        # 2D domain with default units
        self.domain2D1 = modgen2d.discretized_domain2d.DiscretizedDomain2D(
            span_x=5, span_z=4, dx=1, dz=1
        )
    
    def test_default_creator(self):
        interfaces_settings_dict=         {
            'generate_surface':True,
            'rough_interface_creator_instance': 'to_define',
            'rough_interface_generator_scale':[1,2,1,0.2,1,2],   # factor applied to surface interface
            'interfaces_depths_generation':'random', 
            'interfaces_depth_reference_point_x':None, 
            'filter_settings': {
                'filter_window_length':3, 
                'filter_polyorder':2,
            },
            'processing_settings': {
                'simulate_erosion': True,
            }
        }
        
        gen_dict = {'fbm':FBMInterfaceGen(0.6,4,'daviesharte'),
                    'uniform':UniformInterfaceGen(4.5),
                    'normal':UniformInterfaceGen(2)}
        for gen_option in ['fbm', 'normal', 'uniform']:
            interfaces_settings_dict['rough_interface_creator_instance'] = gen_dict[gen_option]
            interface_instance = modgen2d.discretized_interfaces2d_from_dict.DiscretizedInterfaces2DFromDict(
                self.domain2D1, 3, interfaces_settings_dict, rng=np.random.default_rng(2))
        
            self.assertTupleEqual(interface_instance.interfaces_matrix.shape, (7,3))
            
        interfaces_settings_dict['interfaces_depths_generation'] = np.array([0,8,9])
        interfaces_settings_dict['interfaces_depth_reference_point_x']=2.1
        interface_instance = modgen2d.discretized_interfaces2d_from_dict.DiscretizedInterfaces2DFromDict(
                self.domain2D1, 3, interfaces_settings_dict,  rng = np.random.default_rng(2))
        
        self.assertTupleEqual(interface_instance.interfaces_matrix.shape, (7,3))

    def test_no_interface(self): #TODO
        interfaces_settings_dict=         {
            'generator_settings_dict':{
                'generator_option':'uniform',    # options: 'uniform', 'normal', 'fbm'
                'max_dz_per_unit_length':4.5,   # Required for 'uniform'
                'stdev_in_unit_length':2,                         # Required for 'normal'
                'H':0.6, 'length':4, 'method':'daviesharte',  # Required for 'fbm'
            },
            'surface_factor':0.5,                # factor applied to surface interface
            'interfaces_depths_generation':'random', 
            'interfaces_depth_reference_point_x':None, 
            'filter_settings': {
                'filter_window_length':3, 
                'filter_polyorder':2,
            },
            'processing_settings': {
                'prioritize_lower_interface': True,
            }
        }
        for gen_option in ['fbm', 'normal', 'uniform']:
            interfaces_settings_dict['generator_settings_dict']['generator_option'] = gen_option
            A, B = modgen2d.discretized_interfaces2d_from_dict.generate_interfaces_from_interfaces_settings_dict(
                self.domain2D1, 0, interfaces_settings_dict,  np.random.default_rng(2))
        
            self.assertTupleEqual(A.interfaces_matrix.shape, (7,0))
            self.assertTupleEqual(B.interfaces_matrix.shape, (7,1))
            
        interfaces_settings_dict['interfaces_depths_generation'] = np.array([])
        interfaces_settings_dict['interfaces_depth_reference_point_x']=2.1
        A, B = modgen2d.discretized_interfaces2d_from_dict.generate_interfaces_from_interfaces_settings_dict(
                self.domain2D1, 0, interfaces_settings_dict,  np.random.default_rng(2))
        
        self.assertTupleEqual(A.interfaces_matrix.shape, (7,0))
        self.assertTupleEqual(B.interfaces_matrix.shape, (7,1))
        
if __name__ == "__main__":
    unittest.main()
