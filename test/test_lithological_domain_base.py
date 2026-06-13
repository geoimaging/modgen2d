import numpy as np
from .testing_tools import unittest, TestCase

from modgen2d.discretized_domain2d import DiscretizedDomain2D
from modgen2d.lithological_domain2d import LithologicalDomain2DReadOnly

class TestLithologicalDomain2DReadOnly(TestCase):

    def setUp(self):
        self.domain = DiscretizedDomain2D(span_x=5, span_z=4, dx=1, dz=1)

    def test_init(self):
        lit = LithologicalDomain2DReadOnly(self.domain, "base")

        self.assertEqual(lit.name, "base")
        self.assertEqual(lit.lm_type, "NA")
        self.assertIsNone(lit.lithological_matrix)
        self.assertEqual(lit.lit_ids_expected, [])
        self.assertFalse(lit.merged_lit)

    def test_validate_lit_ids_valid(self):
        LithologicalDomain2DReadOnly._validate_lit_ids(["1", "2", "U_1", "X"])

    def test_validate_lit_ids_invalid(self):
        with self.assertRaises(TypeError):
            LithologicalDomain2DReadOnly._validate_lit_ids("1")

        with self.assertRaises(ValueError):
            LithologicalDomain2DReadOnly._validate_lit_ids(["1", "1"])

        with self.assertRaises(TypeError):
            LithologicalDomain2DReadOnly._validate_lit_ids(["1", 2])

        with self.assertRaises(ValueError):
            LithologicalDomain2DReadOnly._validate_lit_ids(["U"])

        with self.assertRaises(ValueError):
            LithologicalDomain2DReadOnly._validate_lit_ids(["U_a"])

    def test_lithological_matrix_valid(self):
        lit = LithologicalDomain2DReadOnly(self.domain, "base")
        lit._set_lit_ids_expected(["1", "2", "U_1", "X"])

        mat = np.array([
            ["1", "1", "2", "2"],
            ["1", "2", "2", "X"],
            ["1", "2", "U_1", "X"],
            ["1", "1", "U_1", "X"],
            ["1", "1", "2", "2"],
        ])

        lit.lithological_matrix = mat

        self.assertArrayEqual(lit.lithological_matrix, mat)
        self.assertDictEqual(
            lit.get_feature_id_and_lit_val(),
            {"def": [1, 2], "U": [1]},
        )

    def test_lithological_matrix_invalid_shape(self):
        lit = LithologicalDomain2DReadOnly(self.domain, "base")
        lit._set_lit_ids_expected(["1"])

        with self.assertRaises(ValueError):
            lit.lithological_matrix = np.array([["1", "1"]])

    def test_lithological_matrix_unexpected_id(self):
        lit = LithologicalDomain2DReadOnly(self.domain, "base")
        lit._set_lit_ids_expected(["1"])

        mat = np.full(self.domain.shape, "2")

        with self.assertRaises(ValueError):
            lit.lithological_matrix = mat

    def test_lithological_matrix_nan_none_invalid(self):
        lit = LithologicalDomain2DReadOnly(self.domain, "base")
        lit._set_lit_ids_expected(["1"])

        mat = np.full(self.domain.shape, "1", dtype=object)
        mat[0, 0] = None

        with self.assertRaises(ValueError):
            lit.lithological_matrix = mat

    def test_check_for_Xs(self):
        self.assertTrue(
            LithologicalDomain2DReadOnly.check_for_Xs(np.array([["1", "X"]]))
        )
        self.assertFalse(
            LithologicalDomain2DReadOnly.check_for_Xs(np.array([["1", "2"]]))
        )

    def test_remeshing_lithological_matrix_replace_false(self):
        lit = LithologicalDomain2DReadOnly(self.domain, "base")
        lit._set_lit_ids_expected(["1", "2"])

        mat = np.array([
            ["1", "1", "2", "2"],
            ["1", "1", "2", "2"],
            ["1", "1", "2", "2"],
            ["1", "1", "2", "2"],
            ["1", "1", "2", "2"],
        ])

        lit.lithological_matrix = mat

        remeshed = lit.remeshing_lithological_matrix(
            new_dx=0.5,
            new_dz=0.5,
            replace=False,
        )

        self.assertTupleEqual(remeshed.domain.shape, (10, 8))
        self.assertTupleEqual(remeshed.lithological_matrix.shape, (10, 8))
        self.assertTupleEqual(lit.domain.shape, (5, 4))

    def test_get_config_and_from_config(self):
        lit = LithologicalDomain2DReadOnly(self.domain, "base")
        lit._set_lit_ids_expected(["1", "2"])

        mat = np.array([
            ["1", "1", "2", "2"],
            ["1", "1", "2", "2"],
            ["1", "1", "2", "2"],
            ["1", "1", "2", "2"],
            ["1", "1", "2", "2"],
        ])

        lit.lithological_matrix = mat

        config = lit.get_config
        recreated = LithologicalDomain2DReadOnly.from_config(config)

        self.assertEqual(recreated.domain, lit.domain)
        self.assertEqual(recreated.name, lit.name)
        self.assertArrayEqual(recreated.lithological_matrix, lit.lithological_matrix)
        self.assertEqual(recreated.lit_ids_expected, lit.lit_ids_expected)

if __name__ == "__main__":
    unittest.main()
