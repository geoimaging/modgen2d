"""Tests for generated_model2d/a_each.py."""

import numpy as np

from .testing_tools import unittest, TestCase
import modgen2d as mg2d
from modgen2d.generated_model2d import GeneratedModel2D
from modgen2d.lithological_domain2d import LithologicalDomain2DReadOnly


def make_lithology(include_x=True):
    length = mg2d.LengthConfig("m", max_grid_density=1000)
    domain = mg2d.DiscretizedDomain2D(1.0, 1.0, 0.25, 0.25, length)
    lithology = LithologicalDomain2DReadOnly(domain, "test")
    ids = ["0", "1", "X"] if include_x else ["0", "1"]
    lithology._set_lit_ids_expected(ids)
    matrix = np.full(domain.shape, "0", dtype="<U1")
    matrix[:, domain.z_centers >= 0.5] = "1"
    if include_x:
        matrix[-1, :] = "X"
    lithology.lithological_matrix = matrix
    lithology.gwt_depth = 0.75
    lithology.lit_order = 0
    return lithology


class TestGeneratedModel2D(TestCase):
    def make_model(self, include_x=True):
        lithology = make_lithology(include_x)
        model = GeneratedModel2D(
            lithology,
            gwt_depth=0.75,
            lit_id2material_dict={"0": ["def", "sand"],
                                  "1": ["def", "clay"],
                                  "extra": ["U", "PVC"]},
        )
        return model

    def test_constructor_filters_extra_material_keys(self):
        model = self.make_model()
        self.assertEqual(set(model.lit_id2material_dict), {"0", "1"})
        self.assertEqual(model.lit_order, 0)
        self.assertEqual(model.gwt_depth, 0.75)

    def test_constructor_rejects_missing_material(self):
        with self.assertRaises(ValueError):
            GeneratedModel2D(
                make_lithology(False), 0.75, {"0": ["def", "sand"]}
            )

    def test_check_accepts_valid_profile_with_ignored_x(self):
        model = self.make_model()
        ignored = model.lit_domain.lithological_matrix == "X"
        profile = np.full(model.lit_domain.domain.shape, 10.0)
        profile[ignored] = model.simulated_val_for_ignored_lit_property
        model.simulated_profiles["su"] = profile
        self.assertIsNone(model.check())

    def test_check_rejects_wrong_shape_nan_and_wrong_ignored_mask(self):
        model = self.make_model()
        model.simulated_profiles["su"] = np.zeros((1, 1))
        with self.assertRaises(ValueError):
            model.check()

        profile = np.zeros(model.lit_domain.domain.shape)
        profile[0, 0] = np.nan
        model.simulated_profiles["su"] = profile
        with self.assertRaises(ValueError):
            model.check()

        model.simulated_profiles["su"] = np.zeros(model.lit_domain.domain.shape)
        with self.assertRaises(ValueError):
            model.check()

    def test_check_can_disallow_ignored_values(self):
        model = self.make_model()
        profile = np.ones(model.lit_domain.domain.shape)
        profile[model.lit_domain.lithological_matrix == "X"] = -99999
        model.simulated_profiles["su"] = profile
        with self.assertRaises(ValueError):
            model.check(allow_ignored_lit_property=False)

    def test_config_round_trip_read_only(self):
        model = self.make_model(include_x=False)
        model.simulated_profiles["su"] = np.arange(
            np.prod(model.lit_domain.domain.shape), dtype=float
        ).reshape(model.lit_domain.domain.shape)
        restored = GeneratedModel2D.from_config(model.get_config, read_only=True)
        self.assertEqual(restored.gwt_depth, model.gwt_depth)
        self.assertEqual(restored.lit_order, model.lit_order)
        self.assertArrayEqual(restored.simulated_profiles["su"],
                              model.simulated_profiles["su"])

    def test_invalid_config_and_missing_plot_property(self):
        with self.assertRaises(TypeError):
            GeneratedModel2D.from_config([])
        with self.assertRaises(ValueError):
            GeneratedModel2D.from_config({})
        with self.assertRaises(ValueError):
            self.make_model().plot_profile("not_generated")


if __name__ == "__main__":
    unittest.main()