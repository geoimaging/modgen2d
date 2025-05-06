# This file is part of <PROJECT> a Python package for <DESCRIPTION>
# Copyright (C) XXXX Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
# LICENSE

"Create basic sanity checks for project."

import geomodgen2d
import numpy as np
import unittest

class TestDomain2D(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.domain2D1 = geomodgen2d.domain2d.Domain2D(span_x=None, span_z=6, del_x=0.2, del_z=2, name='domain_1D')
        cls.domain2D2 = geomodgen2d.domain2d.Domain2D(span_x=5, span_z=4, del_x=0.2, del_z=0.5, name='domain_2D')

    def test_domain2D1_properties(self):
        np.testing.assert_array_equal(self.domain2D1.x_ranges, np.array([0.]))
        np.testing.assert_array_equal(self.domain2D1.z_ranges, np.array([0., 2., 4., 6.]))
        self.assertEqual(self.domain2D1.span_x, 0)
        self.assertEqual(self.domain2D1.del_x, 0)
        self.assertEqual(self.domain2D1.span_z, 6)
        self.assertEqual(self.domain2D1.del_z, 2)

    def test_domain2D2_properties(self):
        np.testing.assert_array_equal(self.domain2D2.x_ranges, np.arange(0, 5.1, 0.2))
        np.testing.assert_array_equal(self.domain2D2.z_ranges, np.array([0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4.]))
        self.assertEqual(self.domain2D2.span_x, 5.)
        self.assertEqual(self.domain2D2.del_x, 0.2)

    def test_remeshing(self):
        remeshed_1D = geomodgen2d.domain2d.check_for_remeshing_coordinate_compatibility(self.domain2D1, None, 6, 1, .25)
        remeshed_2D = geomodgen2d.domain2d.check_for_remeshing_coordinate_compatibility(self.domain2D2, 5, 4, .1, 2)

        np.testing.assert_array_equal(remeshed_1D.x_ranges, np.array([0.]))
        np.testing.assert_array_equal(remeshed_1D.z_ranges, np.arange(0, 6.05, 0.25))
        np.testing.assert_array_equal(remeshed_2D.x_ranges, np.arange(0, 5.05, 0.1))
        np.testing.assert_array_equal(remeshed_2D.z_ranges, np.array([0., 2., 4.]))
        
if __name__ == "__main__":
    unittest.main()
