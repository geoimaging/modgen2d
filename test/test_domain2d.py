# This file is part of <PROJECT> a Python package for <DESCRIPTION>
# Copyright (C) XXXX Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
# LICENSE

"Create basic sanity checks for project."

import numpy as np
from .testing_tools import unittest, TestCase
from modgen2d import discretized_domain2d, length_config

class TestDomain2D(TestCase):
    @classmethod
    def setUpClass(cls):

        # 2D domain with default units
        cls.domain2D1 = discretized_domain2d.DiscretizedDomain2D(
            span_x=5, span_z=4, dx=0.2, dz=0.5
        )
        
        # Custom Units: domain 'l1' with max grid density of 1000000
        length_config_custom = length_config.LengthConfig("l1", max_grid_density=1000000)
        
        cls.length_config_custom = length_config_custom
        cls.domain2D2 = discretized_domain2d.DiscretizedDomain2D(
            span_x=2.200004, span_z=1., dx=0.000004, dz=0.000025, length_config=length_config_custom, 
        )

        cls.domain2D3 = discretized_domain2d.DiscretizedDomain2D(
            span_x=2.200004,
            span_z=1.0,
            dx=0.000004,
            dz=0.000025,
            length_config=length_config_custom,
            origin_x=2.0,
        )
        
    def test_spans_and_dhs(self):
        self.assertEqual(self.domain2D1.spans, [5., 4.])
        self.assertEqual(self.domain2D1.origins, [0.0, 0.0])
        self.assertEqual(self.domain2D1.dhs, [0.2, 0.5])
        
        self.assertEqual(self.domain2D2.spans, [2.200004, 1.])
        self.assertEqual(self.domain2D2.origins, [0.0, 0.0])
        self.assertEqual(self.domain2D2.dhs, [0.000004, 0.000025])
        
        self.assertEqual(self.domain2D2._spans_in_domain_len_units, [2200004, 1000000])
        self.assertEqual(self.domain2D2._origins_in_domain_len_units, [0, 0])
        self.assertEqual(self.domain2D2._dhs_in_domain_len_units, [4, 25])
        
        self.assertEqual(self.domain2D3.spans, [2.200004, 1.0])
        self.assertEqual(self.domain2D3.origins, [2.0, 0.0])
        self.assertEqual(self.domain2D3.dhs, [0.000004, 0.000025])

        self.assertEqual(self.domain2D3._spans_in_domain_len_units, [2200004, 1000000])
        self.assertEqual(self.domain2D3._origins_in_domain_len_units, [2000000, 0])
        self.assertEqual(self.domain2D3._dhs_in_domain_len_units, [4, 25])
        
    def test_xz_centers(self):
        self.assertArrayAlmostEqual(self.domain2D1.x_centers, np.arange(0.1, 5.05, 0.2))
        self.assertArrayAlmostEqual(self.domain2D1.z_centers, np.array([0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75]))
        
        self.assertArrayAlmostEqual(self.domain2D2.x_centers, np.arange(0.000002, 2.200004, 0.000004))
        self.assertArrayAlmostEqual(self.domain2D2.z_centers, np.arange(0.0000125, 1.-0.0000001, 0.000025))

        self.assertArrayAlmostEqual(self.domain2D3.x_centers, np.arange(2.000002, 4.200004, 0.000004))
        self.assertArrayAlmostEqual(self.domain2D3.z_centers, np.arange(0.0000125, 1.-0.0000001, 0.000025))


    def test_get_interface_x_centers_shape(self):
        
        self.assertArrayAlmostEqual(
            self.domain2D1.x_edges,
            np.arange(0.0, 5.2, 0.2),
        )
        self.assertArrayAlmostEqual(
            self.domain2D1.z_edges,
            np.arange(0.0, 4.5, 0.5),
        )
        
        self.assertArrayAlmostEqual(self.domain2D1.get_interface_x_centers, np.arange(-0.1, 5.25, 0.2))
        self.assertArrayAlmostEqual(self.domain2D2.get_interface_x_centers, np.arange(-0.000002, 2.200007, 0.000004))
        self.assertArrayAlmostEqual(self.domain2D3.get_interface_x_centers, np.arange(1.999998, 4.200007, 0.000004))

        
        self.assertTupleEqual(self.domain2D1.shape, (25, 8))
        self.assertTupleEqual(self.domain2D1.interface_shape, (27, 8))
        self.assertTupleEqual(self.domain2D2.shape, (550001, 40000))
        self.assertTupleEqual(self.domain2D2.interface_shape, (550003, 40000))
        self.assertTupleEqual(self.domain2D3.shape, (550001, 40000))
        self.assertTupleEqual(self.domain2D3.interface_shape, (550003, 40000))
        
    def test_remeshing(self):
        remeshed_2D1 = self.domain2D1.remesh(1,2)
        remeshed_2D2 = self.domain2D2.remesh(0.000002, 1)
        remeshed_2D3 = self.domain2D3.remesh(0.000002, 1)

        self.assertArrayEqual(remeshed_2D1.x_centers, [0.5, 1.5, 2.5, 3.5, 4.5])
        self.assertArrayEqual(remeshed_2D1.get_interface_x_centers, [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
        self.assertArrayAlmostEqual(remeshed_2D1.z_centers, [1, 3])
        
        self.assertArrayAlmostEqual(remeshed_2D2.x_centers, np.arange(0.000001, 2.200004, 0.000002))
        self.assertArrayAlmostEqual(remeshed_2D2.get_interface_x_centers, np.arange(-0.000001, 2.200006, 0.000002))
        self.assertArrayAlmostEqual(remeshed_2D2.z_centers, [0.5])
        
        self.assertArrayAlmostEqual(remeshed_2D3.x_centers, np.arange(2.000001, 4.200004, 0.000002))
        self.assertArrayAlmostEqual(remeshed_2D3.get_interface_x_centers, np.arange(-0.000001+2, 4.200006, 0.000002))
        self.assertArrayAlmostEqual(remeshed_2D3.z_centers, [0.5])
        
        self.assertTupleEqual(remeshed_2D1.shape, (5, 2))
        self.assertTupleEqual(remeshed_2D1.interface_shape, (7, 2))
        self.assertTupleEqual(remeshed_2D2.shape, (1100002, 1))
        self.assertTupleEqual(remeshed_2D2.interface_shape, (1100004, 1))
    
    def test_equality(self):
        # Exact same domain should be equal
        same = discretized_domain2d.DiscretizedDomain2D(5, 4., 0.2, 0.5)
        self.assertEqual(self.domain2D1, same)

        same = discretized_domain2d.DiscretizedDomain2D(2.200004, 1., 0.000004, 0.000025, self.length_config_custom)
        self.assertEqual(self.domain2D2, same)
        
        same = discretized_domain2d.DiscretizedDomain2D(2.200004, 1., 0.000004, 0.000025, self.length_config_custom, origin_x=2)
        self.assertEqual(self.domain2D3, same)
        
        # Different dx
        diff_dx = discretized_domain2d.DiscretizedDomain2D(5, 4, 0.1, 0.5)
        self.assertNotEqual(self.domain2D1, diff_dx)
        
        # Different dz
        diff_dz = discretized_domain2d.DiscretizedDomain2D(2.200004, 1., 0.000004, 1, self.length_config_custom)
        self.assertNotEqual(self.domain2D2, diff_dz)
        
        # Different origin
        self.assertNotEqual(self.domain2D2, self.domain2D3)
        self.assertNotEqual(self.domain2D1, self.domain2D2)
        
        # Different units/length_config
        diff_units = discretized_domain2d.DiscretizedDomain2D(5, 4, 0.2, 0.5, self.length_config_custom)
        self.assertNotEqual(self.domain2D1, diff_units)
        
        other_units = length_config.LengthConfig('m', max_grid_density=1000)
        diff_units = discretized_domain2d.DiscretizedDomain2D(5, 4, 0.2, 0.5, other_units)
        self.assertNotEqual(self.domain2D1, diff_units)
    
    def test_is_equivalent(self):
        equivalent = discretized_domain2d.DiscretizedDomain2D(
            2.200004,
            1.0,
            0.000002,
            0.00005,
            self.length_config_custom,
        )

        self.assertTrue(self.domain2D2.is_equivalent(equivalent))
        self.assertFalse(self.domain2D3.is_equivalent(equivalent))
        
    def test_invalid_mesh_error(self):
        # dx does not divide span
        with self.assertRaises(ValueError):
            discretized_domain2d.DiscretizedDomain2D(2.2, 1.0, 0.3, 0.25, self.length_config_custom)
        # dz does not divide span
        with self.assertRaises(ValueError):
            discretized_domain2d.DiscretizedDomain2D(2.2, 1.0, 0.4, 0.2, self.length_config_custom)
            
        # Negative dels or spans
        with self.assertRaises(ValueError):
            discretized_domain2d.DiscretizedDomain2D(-1.0, 1.0, 0.4, 0.2)
        with self.assertRaises(ValueError):
            discretized_domain2d.DiscretizedDomain2D(1.0, -1.0, 0.4, 0.2)
        with self.assertRaises(ValueError):
            discretized_domain2d.DiscretizedDomain2D(0.0, 1.0, 0.4, 0.2)
        with self.assertRaises(ValueError):
            discretized_domain2d.DiscretizedDomain2D(1.0, 0.0, 0.4, 0.2)
        with self.assertRaises(ValueError):
            discretized_domain2d.DiscretizedDomain2D(1.0, 1.0, -0.4, 0.2)        
        with self.assertRaises(ValueError):
            discretized_domain2d.DiscretizedDomain2D(1.0, 0.0, 0.0, 0.2)
        with self.assertRaises(ValueError):
            discretized_domain2d.DiscretizedDomain2D(1.0, 0.0, 0.4, -0.2)
        with self.assertRaises(ValueError):
            discretized_domain2d.DiscretizedDomain2D(2.2, 1.0, 0.4, 0.0)

    def test_get_config_and_from_config(self):
        config = self.domain2D3.get_config
        config_result = {
            "spans_xz": self.domain2D3.spans,
            "dhs_xz": self.domain2D3.dhs,
            "origins_xz": self.domain2D3.origins,
            "length_config":self.domain2D3.length_config.get_config,
        }
        self.assertDictDeepAlmostEqual(config, config_result, tol=1e-6)

        recreated = discretized_domain2d.DiscretizedDomain2D.from_config(config_result)
        self.assertEqual(self.domain2D3, recreated)
        
    def test_can_domain_be_remeshed(self):
        # Valid remesh
        self.assertTrue(self.domain2D1.can_domain_be_remeshed(0.25, 0.2))
        # Invalid remesh
        self.assertFalse(self.domain2D1.can_domain_be_remeshed(0.3, 0.25))
        self.assertFalse(self.domain2D1.can_domain_be_remeshed(0.2, 0.15))
        
if __name__ == "__main__":
    unittest.main()
