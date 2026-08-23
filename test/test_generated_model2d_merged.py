"""Tests for generated_model2d/a_merged.py."""

import warnings

from .testing_tools import unittest, TestCase
from modgen2d.generated_model2d import (
    GeneratedModel2DMerged,
    GeneratedProfileCollection2D,
)
from modgen2d.spatial_simulator2d import ConstantSimulator


class TestGeneratedModel2DMerged(TestCase):
    def test_invalid_collection_uses_documented_failure_state(self):
        """A failed merge warns and leaves all computed fields unavailable."""
        collection = GeneratedProfileCollection2D.__new__(
                    GeneratedProfileCollection2D
                )
        collection._spatial_simulator2d_instance = ConstantSimulator()
                
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            merged = GeneratedModel2DMerged(collection)

        self.assertIsNone(merged.lit_domain)
        self.assertIsNone(merged.simulated_profiles)
        self.assertIsNone(merged.lit_id2material_dict)
        self.assertIsNone(merged.gwt_depth)
        self.assertEqual(merged.lit_order, -1)
        self.assertTrue(any("Failed to merge lithological domains" in str(w.message)
                            for w in caught))

    def test_merged_model_starts_unlocked(self):
        collection = GeneratedProfileCollection2D.__new__(
            GeneratedProfileCollection2D
        )
        collection._spatial_simulator2d_instance = ConstantSimulator()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            merged = GeneratedModel2DMerged(collection)

        self.assertFalse(merged._locked)

if __name__ == "__main__":
    unittest.main()