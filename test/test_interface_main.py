# This file is part of <PROJECT> a Python package for <DESCRIPTION>
# Copyright (C) XXXX Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
# LICENSE

"Create basic sanity checks for project."

import modgen2d
import numpy as np
from .testing_tools import unittest, TestCase

from modgen2d.interface import DiscretizedInterfaces2D
from modgen2d.interface.rough_interface_generator import UniformInterfaceGen
from modgen2d.interface.interface_smoother import SavGol2DSmoother
from modgen2d.interface.depth_updaters import EquidistantDepthUpdater


class TestDiscretizedInterfaces2D(TestCase):

    def setUp(self):
        self.domain = modgen2d.discretized_domain2d.DiscretizedDomain2D(
            span_x=5, span_z=4, dx=1, dz=1
        )
        self.interfaces = DiscretizedInterfaces2D(
            self.domain, n_soil_layers=3, generate_surface=True,
            rng=np.random.default_rng(2)
        )

    def test_init(self):
        self.assertTupleEqual(self.interfaces.shape, (7, 3))
        self.assertTrue(np.isnan(self.interfaces.interfaces_matrix).all())
        self.assertFalse(self.interfaces._locked)
        self.assertEqual(self.interfaces.n_soil_layers, 3)
        self.assertTrue(self.interfaces.generate_surface)

    def test_set_interfaces_matrix(self):
        mat = np.array([
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
        ], dtype=float)

        self.interfaces.set_interfaces_matrix(mat)
        self.assertArrayEqual(self.interfaces.interfaces_matrix, mat)

    def test_set_interfaces_matrix_invalid(self):
        with self.assertRaises(ValueError):
            self.interfaces.set_interfaces_matrix(np.ones((7, 2)))

        with self.assertRaises(ValueError):
            self.interfaces.set_interfaces_matrix(np.ones((6, 3)))

        bad = np.ones((7, 3))
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.interfaces.set_interfaces_matrix(bad)

    def test_apply_pipeline_steps(self):
        gen = UniformInterfaceGen(
            max_dz_per_unit_length=1.0,
            generate_surface=True,
            roughness_multipliers=[1, 1, 1],
        )
        smoother = SavGol2DSmoother(3, 1)
        updater = EquidistantDepthUpdater()

        out = self.interfaces.apply(gen).apply(smoother).apply(updater)

        self.assertIs(out, self.interfaces)
        self.assertTupleEqual(self.interfaces.interfaces_matrix.shape, (7, 3))
        self.assertFalse(np.isnan(self.interfaces.interfaces_matrix).any())

    def test_overlap_resolution_erosion(self):
        mat = np.array([
            [1, 1.7, 1.5],
            [0.5, 1, 3],
            [0, 2, 5],
            [3, 4, 2],
            [3, 4, 2],
            [3, 4, 2],
            [2.2, 2, 2.5],
        ], dtype=float)

        self.interfaces.set_interfaces_matrix(mat)
        overlap, _ = self.interfaces.check_if_overlapping_interfaces()
        self.assertTrue(overlap)

        self.interfaces.apply("erosion")

        overlap, _ = self.interfaces.check_if_overlapping_interfaces()
        self.assertFalse(overlap)
        self.assertTrue(np.all(np.diff(self.interfaces.interfaces_matrix, axis=1) >= 0))

    def test_adjust_surface_top_to_zero(self):
        mat = np.array([
            [1, 2, 3],
            [0.5, 2, 3],
            [-1, 1, 2],
            [0, 2, 4],
            [1, 3, 5],
            [2, 4, 6],
            [1, 3, 4],
        ], dtype=float)

        self.interfaces.set_interfaces_matrix(mat)
        self.interfaces.apply("adjust_surface_top_to_zero")

        self.assertAlmostEqual(np.min(self.interfaces.interfaces_matrix[:, 0]), 0.0)

    def test_lock_interfaces(self):
        mat = np.tile([0, 1, 2], (7, 1)).astype(float)
        self.interfaces.set_interfaces_matrix(mat)

        self.interfaces.lock_interfaces()
        self.assertTrue(self.interfaces._locked)

        with self.assertRaises(SystemError):
            self.interfaces.set_interfaces_matrix(mat)

    def test_remesh_interface(self):
        mat = np.tile([0, 1, 2], (7, 1)).astype(float)
        self.interfaces.set_interfaces_matrix(mat)

        remeshed = self.interfaces.remesh_interface(new_dx=0.5)

        self.assertTupleEqual(remeshed.interfaces_matrix.shape, (12, 3))
        self.assertTupleEqual(remeshed.domain.shape, (10, 4))

    def test_invalid_apply_command(self):
        with self.assertRaises(ValueError):
            self.interfaces.apply("bad_command")     
        
if __name__ == "__main__":
    unittest.main()
