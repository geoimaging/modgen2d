"""Tests for generated_model2d/b_collection_main.py."""

import tempfile
import warnings

import h5py
import numpy as np

from .testing_tools import unittest, TestCase
import modgen2d as mg2d
from modgen2d.generated_model2d import GeneratedModel2D, GeneratedProfileCollection2D
from modgen2d.generated_model2d.b_collection_main import (
    convert_string_array_for_hdf5,
    save_dict_to_hdf5,
)
from modgen2d.lithological_domain2d import LithologicalDomain2DReadOnly
from modgen2d.spatial_simulator2d import CovarianceDecompositionSimulator


def make_lithology(order=0):
    length = mg2d.LengthConfig("m", max_grid_density=1000)
    domain = mg2d.DiscretizedDomain2D(1.0, 1.0, 0.25, 0.25, length)
    lithology = LithologicalDomain2DReadOnly(domain, f"set_{order}")
    lithology._set_lit_ids_expected(["0"])
    lithology.lithological_matrix = np.full(domain.shape, "0", dtype="<U1")
    lithology.lit_order = order
    lithology.gwt_depth = None
    return lithology


def make_generated_model(order=0, simulated_profiles=None):
    """Create a genuine GeneratedModel2D with only its required state."""
    model = GeneratedModel2D(
        make_lithology(order),
        gwt_depth=None,
        lit_id2material_dict={"0": ["def", "soil"]},
    )
    model.simulated_profiles = (
        {} if simulated_profiles is None else simulated_profiles
    )
    return model


def make_workflow_collection(seed=42):
    """Build the minimum genuine collection needed for property generation."""
    material_dict = {"0": ["def", "soil"]}
    lithology = make_lithology(order=0)
    merged_lithology = LithologicalDomain2DReadOnly.from_config(
        lithology.get_config
    )
    merged_lithology.lit_order = -1

    collection = GeneratedProfileCollection2D.__new__(
        GeneratedProfileCollection2D
    )
    collection._lit_id2material_dict = material_dict
    collection._sampled_properties = {
        "property_a": {
            "0": {"both": {
                "mean": 10.0,
                "mean_slope_with_depth": 0.0,
                "stdev_or_cov": 2.0,
                "stdev_type": "stdev",
            }}
        },
        "property_b": {
            "0": {"both": {
                "mean": 100.0,
                "mean_slope_with_depth": 0.0,
                "stdev_or_cov": 5.0,
                "stdev_type": "stdev",
            }}
        },
    }
    collection._main_properties_unique_code = 1
    collection._generated_properties_list = []
    collection._read_only = False
    collection._spatial_simulator2d_instance = (
        CovarianceDecompositionSimulator(
            theta_x=1.0,
            theta_z=0.5,
            rng=np.random.default_rng(seed),
        )
    )
    collection._generated_model2d_set = {
        "soil": GeneratedModel2D(
            lithology,
            gwt_depth=None,
            lit_id2material_dict=material_dict,
        )
    }
    collection._merged_generated_model2d = GeneratedModel2D(
        merged_lithology,
        gwt_depth=None,
        lit_id2material_dict=material_dict,
    )
    return collection


class TestGeneratedProfileCollection2D(TestCase):
    def test_shared_zvals_generate_correlated_property_profiles(self):
        """The collection must reuse one stored field for both properties."""
        collection = make_workflow_collection(seed=101)
        collection.simulate_zvals_property_profile("shared_z")

        shared_z = (
            collection.generated_model2d_set["soil"]
            .simulated_profiles["shared_z"]
            .copy()
        )
        collection.simulate_profile_from_zvals_property_profile(
            "property_a", "shared_z"
        )
        collection.simulate_profile_from_zvals_property_profile(
            "property_b", "shared_z"
        )

        soil_profiles = collection.generated_model2d_set["soil"].simulated_profiles
        merged_profiles = collection.merged_generated_model2d.simulated_profiles
        property_a = soil_profiles["property_a"]
        property_b = soil_profiles["property_b"]

        standardized_a = (property_a - 10.0) / 2.0
        standardized_b = (property_b - 100.0) / 5.0
        self.assertArrayAlmostEqual(standardized_a, shared_z)
        self.assertArrayAlmostEqual(standardized_b, shared_z)
        self.assertArrayAlmostEqual(standardized_a, standardized_b)
        self.assertAlmostEqual(
            np.corrcoef(property_a.ravel(), property_b.ravel())[0, 1],
            1.0,
        )
        self.assertArrayEqual(merged_profiles["property_a"], property_a)
        self.assertArrayEqual(merged_profiles["property_b"], property_b)

    def test_simulate_property_profile_stores_output_and_deletes_temporary_zvals(self):
        """The convenience workflow must leave only the requested property."""
        collection = make_workflow_collection(seed=202)
        collection.simulate_property_profile("property_a")

        soil_profiles = collection.generated_model2d_set["soil"].simulated_profiles
        merged_profiles = collection.merged_generated_model2d.simulated_profiles
        self.assertIn("property_a", soil_profiles)
        self.assertIn("property_a", merged_profiles)
        self.assertEqual(
            soil_profiles["property_a"].shape,
            collection.generated_model2d_set["soil"].lit_domain.domain.shape,
        )
        self.assertFalse(
            any(name.startswith("___z_vals_") for name in soil_profiles)
        )
        self.assertFalse(
            any(name.startswith("___z_vals_") for name in merged_profiles)
        )

    def test_clip_preserves_sentinel_and_clips_valid_values(self):
        profile = np.array([[-99999.0, -2.0, 3.0, 12.0]])
        result = GeneratedProfileCollection2D.clip_simulated_profile(
            profile, min_val=0.0, max_val=10.0,
            simulated_val_for_ignored_lit_property=-99999,
        )
        self.assertArrayEqual(result, [[-99999.0, 0.0, 3.0, 10.0]])

    def test_clip_warning_error_and_input_validation(self):
        profile = np.array([-1.0, 5.0, 11.0])
        with self.assertWarns(UserWarning):
            GeneratedProfileCollection2D.clip_simulated_profile(
                profile, 0.0, 10.0, warn=True
            )
        with self.assertRaises(ValueError):
            GeneratedProfileCollection2D.clip_simulated_profile(
                profile, 0.0, 10.0, raise_error=True
            )
        with self.assertRaises(AssertionError):
            GeneratedProfileCollection2D.clip_simulated_profile(
                profile, min_val="zero"
            )

    def test_merge_profile_initializes_then_overwrites_nonignored_cells(self):
        lithology = make_lithology()
        base = np.full(lithology.domain.shape, 1.0)
        overlay = np.full(lithology.domain.shape, -99999.0)
        overlay[0, 0] = 9.0

        initialized = GeneratedProfileCollection2D.get_merged_simulated_profile(
            lithology, None, lithology, base, -99999
        )
        merged = GeneratedProfileCollection2D.get_merged_simulated_profile(
            lithology, initialized, lithology, overlay, -99999
        )
        expected = base.copy()
        expected[0, 0] = 9.0
        self.assertArrayEqual(merged, expected)

    def test_ordering_and_duplicate_order_validation(self):
        collection = GeneratedProfileCollection2D.__new__(GeneratedProfileCollection2D)
        collection._generated_model2d_set = {
            "later": make_generated_model(1),
            "first": make_generated_model(0),
        }
        ordered, domains = collection.get_ordered_set_names_dict()
        self.assertEqual(ordered, {0: "first", 1: "later"})
        self.assertEqual(set(domains), {"first", "later"})

        collection._generated_model2d_set["duplicate"] = make_generated_model(1)
        with self.assertRaises(ValueError):
            collection.get_ordered_set_names_dict()

    def test_getters_and_delete_property(self):
        collection = GeneratedProfileCollection2D.__new__(GeneratedProfileCollection2D)
        first = make_generated_model(
            0, {"su": np.ones(make_lithology(0).domain.shape)}
        )
        merged = make_generated_model(
            -1, {"su": np.ones(make_lithology(-1).domain.shape)}
        )
        collection._generated_model2d_set = {"soil": first}
        collection._merged_generated_model2d = merged

        self.assertIs(collection.get_generated_model2d("soil"), first)
        self.assertIs(collection.get_generated_model2d(None), merged)
        self.assertEqual(set(collection.get_simulated_properties), {"su"})
        collection.delete_simulated_property_profile("su")
        self.assertEqual(first.simulated_profiles, {})
        self.assertEqual(merged.simulated_profiles, {})
        with self.assertRaises(ValueError):
            collection.delete_simulated_property_profile("su")

 
if __name__ == "__main__":
    unittest.main()