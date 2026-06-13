# This file is part of <PROJECT> a Python package for <DESCRIPTION>
# Copyright (C) XXXX Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
# LICENSE

"Create basic sanity checks for project."

import numpy as np
import modgen2d
from .testing_tools import unittest, TestCase

from modgen2d.interface import DiscretizedInterfaces2DFromDict
from modgen2d.interface.rough_interface_generator import UniformInterfaceGen


class TestDiscretizedInterfaces2DFromDict(TestCase):

    def setUp(self):
        self.domain = modgen2d.discretized_domain2d.DiscretizedDomain2D(
            span_x=5, span_z=4, dx=1, dz=1
        )

    def get_base_settings(self):
        return {
            "generate_surface": True,
            "rough_interface_generator_instance": UniformInterfaceGen(
                1.5, True, [1, 1, 1]
            ),
            "savgol2d_smoother_settings": {
                "filter_window_length": 3,
                "filter_polyorder": 1,
            },
            "interfaces_depths_updater": "equidistant",
            "interfaces_depth_reference_point_x": None,
            "overlapping_resolver_technique": "erosion",
            "adjust_surface_top_to_zero": True,
        }

    def test_from_dict_valid(self):
        obj = DiscretizedInterfaces2DFromDict(
            self.domain, 3, self.get_base_settings(),
            rng=np.random.default_rng(2)
        )

        self.assertTupleEqual(obj.interfaces_matrix.shape, (7, 3))
        self.assertFalse(np.isnan(obj.interfaces_matrix).any())
        self.assertTrue(obj._locked)

        overlap, _ = obj.check_if_overlapping_interfaces()
        self.assertFalse(overlap)

    def test_from_dict_manual_depths(self):
        settings = self.get_base_settings()
        settings["interfaces_depths_updater"] = np.array([0.0, 1.5, 3.0])
        settings["interfaces_depth_reference_point_x"] = 2.5

        obj = DiscretizedInterfaces2DFromDict(
            self.domain, 3, settings,
            rng=np.random.default_rng(2)
        )

        self.assertTupleEqual(obj.interfaces_matrix.shape, (7, 3))
        self.assertEqual(obj._ref_x, 2.5)

    def test_from_dict_no_surface(self):
        settings = {
            "generate_surface": False,
            "rough_interface_generator_instance": UniformInterfaceGen(
                1.5, False, [0, 1, 1]
            ),
            "savgol2d_smoother_settings": {
                "filter_window_length": 3,
                "filter_polyorder": 1,
            },
            "interfaces_depths_updater": "equidistant",
            "interfaces_depth_reference_point_x": None,
            "overlapping_resolver_technique": "erosion",
            "adjust_surface_top_to_zero": False,
        }

        obj = DiscretizedInterfaces2DFromDict(
            self.domain, 3, settings,
            rng=np.random.default_rng(2)
        )

        self.assertTupleEqual(obj.interfaces_matrix.shape, (7, 3))
        self.assertTrue(np.allclose(obj.interfaces_matrix[:, 0], 0.0))

    def test_missing_required_key(self):
        settings = self.get_base_settings()
        del settings["interfaces_depth_reference_point_x"]

        with self.assertRaises(KeyError):
            DiscretizedInterfaces2DFromDict(
                self.domain, 3, settings,
                rng=np.random.default_rng(2)
            )

    def test_unknown_key(self):
        settings = self.get_base_settings()
        settings["bad_key"] = 123

        with self.assertRaises(KeyError):
            DiscretizedInterfaces2DFromDict(
                self.domain, 3, settings,
                rng=np.random.default_rng(2)
            )

    def test_surface_flag_mismatch(self):
        settings = self.get_base_settings()
        settings["rough_interface_generator_instance"] = UniformInterfaceGen(
            1.5, False, [0, 1, 1]
        )

        with self.assertRaises(ValueError):
            DiscretizedInterfaces2DFromDict(
                self.domain, 3, settings,
                rng=np.random.default_rng(2)
            )
        
if __name__ == "__main__":
    unittest.main()
