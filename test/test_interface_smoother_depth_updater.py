# This file is part of <PROJECT> a Python package for <DESCRIPTION>
# Copyright (C) XXXX Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
# LICENSE

"Create basic sanity checks for project."

import modgen2d
import numpy as np
from .testing_tools import unittest, TestCase

from modgen2d.interface import DiscretizedInterfaces2D
from modgen2d.interface import DiscretizedInterfaces2D
from modgen2d.interface.interface_smoother import SavGol2DSmoother
from modgen2d.interface.depth_updaters import (
    AbstractDepthUpdater,
    OneBoreholeDepthUpdater,
    EquidistantDepthUpdater,
    RandomDepthUpdater,
)

class TestInterfaceSmoothers(TestCase):

    def setUp(self):
        domain = modgen2d.discretized_domain2d.DiscretizedDomain2D(
            span_x=5, span_z=4, dx=1, dz=1
        )
        self.interfaces = DiscretizedInterfaces2D(
            domain, 3, True, rng=np.random.default_rng(2)
        )
        self.interfaces.set_interfaces_matrix(np.zeros((7, 3)))


    def test_savgol_smoother(self):
        mat = np.array([
            [2, 0, 0],
            [3, 1, 1],
            [2, 4, 6],
            [1, 6, 7],
            [2, 7, 5],
            [3, 2, 1],
            [4, 5, 3],
        ], dtype=float)

        self.interfaces.set_interfaces_matrix(mat)

        smoother = SavGol2DSmoother(3, 1)
        out = smoother.generate_smooth_interfaces(self.interfaces)

        self.assertTupleEqual(out.shape, (7, 3))
        self.assertFalse(np.isnan(out).any())

    def test_apply_savgol_smoother(self):
        mat = np.tile([0, 1, 2], (7, 1)).astype(float)
        self.interfaces.set_interfaces_matrix(mat)

        smoother = SavGol2DSmoother(3, 1)
        self.interfaces.apply(smoother)

        self.assertTupleEqual(self.interfaces.interfaces_matrix.shape, (7, 3))
        
    # Depth Updaters
    def test_adjust_ref_x_none(self):
        ref_x = AbstractDepthUpdater.adjust_ref_x(None, self.interfaces)
        self.assertEqual(ref_x, -0.5)

    def test_adjust_ref_x_outside_warns(self):
        with self.assertWarns(UserWarning):
            ref_x = AbstractDepthUpdater.adjust_ref_x(-5, self.interfaces)

        self.assertEqual(ref_x, -0.5)

    def test_adjust_ref_x_invalid(self):
        with self.assertRaises(ValueError):
            AbstractDepthUpdater.adjust_ref_x("bad", self.interfaces)

    def test_one_borehole_depth_updater(self):
        updater = OneBoreholeDepthUpdater([0, 1.5, 3.0], ref_x=2.5)
        mat, ref_x = updater.update_interfaces_depths(self.interfaces)

        self.assertTupleEqual(mat.shape, (7, 3))
        self.assertEqual(ref_x, 2.5)
        self.assertArrayAlmostEqual(mat[3, :], [0, 1.5, 3.0])

    def test_one_borehole_invalids(self):
        with self.assertRaises(ValueError):
            OneBoreholeDepthUpdater(None).update_interfaces_depths(self.interfaces)

        with self.assertRaises(ValueError):
            OneBoreholeDepthUpdater([1, 2, 3]).update_interfaces_depths(self.interfaces)

        with self.assertRaises(ValueError):
            OneBoreholeDepthUpdater([0, 3, 2]).update_interfaces_depths(self.interfaces)

        with self.assertRaises(ValueError):
            OneBoreholeDepthUpdater([0, 1]).update_interfaces_depths(self.interfaces)

    def test_equidistant_depth_updater(self):
        updater = EquidistantDepthUpdater()
        mat, ref_x = updater.update_interfaces_depths(self.interfaces)

        self.assertTupleEqual(mat.shape, (7, 3))
        self.assertEqual(ref_x, -0.5)
        self.assertArrayAlmostEqual(mat[0, :], [0, 4 / 3, 8 / 3])

    def test_random_depth_updater(self):
        updater = RandomDepthUpdater()
        mat, ref_x = updater.update_interfaces_depths(self.interfaces)

        self.assertTupleEqual(mat.shape, (7, 3))
        self.assertEqual(ref_x, -0.5)
        self.assertTrue(np.all(np.diff(mat, axis=1) >= 0))

if __name__ == "__main__":
    unittest.main()
