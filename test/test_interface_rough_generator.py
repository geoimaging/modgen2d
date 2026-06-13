# This file is part of <PROJECT> a Python package for <DESCRIPTION>
# Copyright (C) XXXX Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
# LICENSE

"Create basic sanity checks for project."

import modgen2d
import numpy as np
from .testing_tools import unittest, TestCase

from modgen2d.interface import DiscretizedInterfaces2D
from modgen2d.interface.rough_interface_generator import (
    AbstractRoughInterfaceGenerator,
    UniformInterfaceGen,
    NormalInterfaceGen,
    # FBMInterfaceGen,
)

class TestRoughInterfaceGenerators(TestCase):

    def setUp(self):
        self.domain = modgen2d.discretized_domain2d.DiscretizedDomain2D(
            span_x=5, span_z=4, dx=1, dz=1
        )
        self.interfaces = DiscretizedInterfaces2D(
            self.domain, 3, True, rng=np.random.default_rng(2)
        )

    def test_uniform_generator(self):
        gen = UniformInterfaceGen(1.5, True, [1, 1, 1])
        mat, multipliers = gen.generate_rough_interfaces(self.interfaces)

        self.assertTupleEqual(mat.shape, (7, 3))
        self.assertArrayAlmostEqual(multipliers, [1, 1, 1])
        self.assertFalse(np.isnan(mat).any())

        dz_per_dx = np.abs(np.diff(mat, axis=0)) / self.domain.dhs[0]
        self.assertTrue(np.all(dz_per_dx <= 1.5 + 1e-12))

    def test_normal_generator(self):
        gen = NormalInterfaceGen(0.2, True, [1, 1, 1])
        mat, multipliers = gen.generate_rough_interfaces(self.interfaces)

        self.assertTupleEqual(mat.shape, (7, 3))
        self.assertArrayAlmostEqual(multipliers, [1, 1, 1])
        self.assertGreater(np.std(mat), 0.0)

    # def test_fbm_generator(self):
    #     gen = FBMInterfaceGen(0.75, 5, "daviesharte", True, [1, 1, 1])
    #     mat, multipliers = gen.generate_rough_interfaces(self.interfaces)

    #     self.assertTupleEqual(mat.shape, (7, 3))
    #     self.assertArrayAlmostEqual(multipliers, [1, 1, 1])
    #     self.assertGreater(np.std(mat), 0.0)

    def test_adjusted_roughness_multipliers_expand(self):
        gen = UniformInterfaceGen(1.0, True, [1, 2])
        out = gen.get_adjusted_roughness_multipliers(self.interfaces, gen.roughness_multipliers)

        self.assertArrayAlmostEqual(out, [1, 2, 2])

    def test_surface_flag_mismatch(self):
        gen = UniformInterfaceGen(1.0, False, [0, 1, 1])

        with self.assertRaises(ValueError):
            gen.generate_rough_interfaces(self.interfaces)

    def test_invalid_roughness_multipliers(self):
        with self.assertRaises(ValueError):
            AbstractRoughInterfaceGenerator.check_roughness_multipliers(
                np.array([1, 1]), generate_surface=False
            )

        with self.assertRaises(ValueError):
            AbstractRoughInterfaceGenerator.check_roughness_multipliers(
                np.array([0, 1]), generate_surface=True
            )

        with self.assertRaises(ValueError):
            AbstractRoughInterfaceGenerator.check_roughness_multipliers(
                np.array([]), generate_surface=True
            )

if __name__ == "__main__":
    unittest.main()
