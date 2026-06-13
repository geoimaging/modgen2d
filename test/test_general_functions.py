import numpy as np
from .testing_tools import unittest, TestCase
import modgen2d.general_functions as f

class TestGeneralFunctions(TestCase):
    
    def test_is_divisible(self):
        self.assertRaises(ValueError, f.is_divisible, 0, 1) 
        self.assertRaises(ValueError, f.is_divisible, 2, 2.1) 
        self.assertRaises(ValueError, f.is_divisible, -1, 2) 
        self.assertRaises(ValueError, f.is_divisible, 1, -2)
        self.assertRaises(TypeError, f.is_divisible, None, -2)
        self.assertFalse(f.is_divisible(1,0)) 
        self.assertFalse(f.is_divisible(3.1,2)) 
        self.assertTrue(f.is_divisible(4.,2)) 
        self.assertFalse(f.is_divisible(4.000001, 2))
        self.assertTrue(f.is_divisible(4.000001, 2, tol=1e-5))
    
    def test_check_integer(self):
        self.assertRaises(ValueError, f.check_integer, 2.01)
        self.assertRaises(ValueError, f.check_integer, -2.01)
        self.assertRaises(TypeError, f.check_integer, None)
        self.assertRaises(ValueError, f.check_integer, (4.000001))
        self.assertEqual(f.check_integer(4.0000000001), 4)
        self.assertEqual(f.check_integer(4.), 4)
        self.assertEqual(f.check_integer(-2.), -2)
    
    def test_is_integer_value(self):
        self.assertRaises(TypeError, f.is_integer_value, None)
        self.assertTrue(f.is_integer_value(2.))
        self.assertTrue(f.is_integer_value(-2))
        self.assertTrue(f.is_integer_value('2'))
        self.assertTrue(f.is_integer_value('0'))
        self.assertFalse(f.is_integer_value(2.1))
        self.assertFalse(f.is_integer_value(-2.000001))
        self.assertFalse(f.is_integer_value('2f'))
        
    def test_is_close(self):
        self.assertFalse(f.is_close(2.0000001, 2.00004))
        self.assertTrue(f.is_close(2.0000001, 2.00004, 1e-3))
        self.assertFalse(f.is_close(10, 1000))
    
    def test_check_obstruction_id_valid(self):
        self.assertEqual(f.check_obstruction_id(1), 1)
        self.assertEqual(f.check_obstruction_id(1.0), 1)
        self.assertEqual(f.check_obstruction_id("2.0"), 2)
        
    def test_check_obstruction_id_invalid(self):
        self.assertEqual(f.check_obstruction_id(2),2)
        self.assertEqual(f.check_obstruction_id(2.00),2)
        self.assertRaises(ValueError, f.check_obstruction_id, 0) 
        self.assertRaises(ValueError, f.check_obstruction_id, 0.3) 
        self.assertRaises(ValueError, f.check_obstruction_id, -1) 
        
    def test_is_valid_feature_id(self):
        valid, _ = f.is_valid_feature_id("ABA")
        self.assertTrue(valid)

        valid, _ = f.is_valid_feature_id("soil")
        self.assertTrue(valid)
        
        invalid_values = [
            None,
            "",
            "def",
            "AQWEQWEQE",
            "A8A",
            "A_",
            "_V",
            "1",
            ["A"],
            [12,23],
        ]

        for val in invalid_values:
            valid, _ = f.is_valid_feature_id(val)
            self.assertFalse(valid)
            
    def test_validate_feature_ids_list_valid(self):
        f.validate_feature_ids_list(["def"])
        f.validate_feature_ids_list(["def", "util", "rock"])

    def test_validate_feature_ids_list_invalid(self):
        with self.assertRaises(KeyError):
            f.validate_feature_ids_list(["soil"])


        with self.assertRaises(KeyError):
            f.validate_feature_ids_list(["def", "bad_"])

        with self.assertRaises(KeyError):
            f.validate_feature_ids_list(["def", "bad1"])

    def test_format_lith_ids(self):
        self.assertEqual(
            f.format_lith_ids("def", ["sand", "clay"]),
            ["sand", "clay"],
        )

        self.assertEqual(
            f.format_lith_ids("util", ["metal", "plastic"]),
            ["util_metal", "util_plastic"],
        )

    def test_format_value(self):
        self.assertEqual(f.format_value(2), "2")
        self.assertEqual(f.format_value(2.0), "2")
        self.assertEqual(f.format_value("2.0"), "2")
        self.assertEqual(f.format_value(2.5), "2.5")
        self.assertEqual(f.format_value("sand"), "sand")

    def test_remeshing_2D_matrix_same_grid(self):
        x = np.array([0, 1, 2])
        z = np.array([0, 1])
        mat = np.array([
            [1, 2],
            [3, 4],
            [5, 6],
        ])

        out = f.remeshing_2D_matrix(x, x, z, z, mat)

        self.assertArrayEqual(out, mat)

    def test_remeshing_2D_matrix_nearest(self):
        x_old = np.array([0, 1])
        z_old = np.array([0, 1])
        mat = np.array([
            [1, 2],
            [3, 4],
        ])

        x_new = np.array([0, 0.9])
        z_new = np.array([0, 0.9])

        out = f.remeshing_2D_matrix(
            x_old,
            x_new,
            z_old,
            z_new,
            mat,
            interp_method="nearest",
        )

        expected = np.array([
            [1, 2],
            [3, 4],
        ])

        self.assertArrayAlmostEqual(out, expected)

    def test_remeshing_2D_matrix_linear(self):
        x_old = np.array([0, 1])
        z_old = np.array([0, 1])
        mat = np.array([
            [0, 1],
            [1, 2],
        ], dtype=float)

        x_new = np.array([0.5])
        z_new = np.array([0.5])

        out = f.remeshing_2D_matrix(
            x_old,
            x_new,
            z_old,
            z_new,
            mat,
            interp_method="linear",
        )

        self.assertArrayAlmostEqual(out, np.array([[1.0]]))

    def test_fixed_key_dict_valid_update(self):
        d = f.FixedKeyDict({
            "a": 1,
            "b": {
                "c": 2,
            },
        })

        d["a"] = 10
        d["b"] = {"c": 20}

        self.assertEqual(d["a"], 10)
        self.assertEqual(d["b"]["c"], 20)

    def test_fixed_key_dict_invalid_update(self):
        d = f.FixedKeyDict({
            "a": 1,
            "b": {
                "c": 2,
            },
        })

        with self.assertRaises(KeyError):
            d["new"] = 5

        with self.assertRaises(TypeError):
            d["b"] = 5

    def test_validate_processed_property_dict_both(self):
        prop = {
            "L1": {
                "both": {
                    "mean": 5,
                    "stdev_or_cov": 1.0,
                    "stdev_type": "stdev",
                }
            }
        }

        out = f.validate_processed_property_dict(prop)

        self.assertEqual(out["L1"]["both"]["mean_slope_with_depth"], 0.0)

    def test_validate_processed_property_dict_wet_dry(self):
        prop = {
            "L1": {
                "wet": {
                    "mean": 10,
                    "stdev_or_cov": 0.2,
                    "stdev_type": "cov",
                },
                "dry": {
                    "mean": 8,
                    "mean_slope_with_depth": 1.5,
                    "stdev_or_cov": 1.5,
                    "stdev_type": "stdev",
                },
            }
        }

        out = f.validate_processed_property_dict(prop)

        self.assertEqual(out["L1"]["wet"]["mean_slope_with_depth"], 0.0)
        self.assertEqual(out["L1"]["dry"]["mean_slope_with_depth"], 1.5)

    def test_validate_processed_property_dict_invalid(self):
        with self.assertRaises(AssertionError):
            f.validate_processed_property_dict(["not", "a", "dict"])

        with self.assertRaises(AssertionError):
            f.validate_processed_property_dict({
                "L1": {
                    "both": {
                        "stdev_or_cov": 1.0,
                        "stdev_type": "stdev",
                    }
                }
            })

        with self.assertRaises(AssertionError):
            f.validate_processed_property_dict({
                "L1": {
                    "both": {
                        "mean": 5,
                        "stdev_or_cov": 1.0,
                        "stdev_type": "variance",
                    }
                }
            })

        with self.assertRaises(AssertionError):
            f.validate_processed_property_dict({
                "L1": {
                    "wet": {
                        "mean": 5,
                        "stdev_or_cov": 1.0,
                        "stdev_type": "stdev",
                    }
                }
            })

    def test_deep_object_equivalent_scalars(self):
        self.assertTrue(f.deep_object_equivalent(1, 1))
        self.assertFalse(f.deep_object_equivalent(1, 2))
        self.assertTrue(f.deep_object_equivalent(1.0, 1.0 + 1e-13))

    def test_deep_object_equivalent_arrays(self):
        self.assertTrue(
            f.deep_object_equivalent(
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
            )
        )

        self.assertFalse(
            f.deep_object_equivalent(
                np.array([1.0, 2.0]),
                np.array([1.0, 3.0]),
            )
        )

    def test_deep_object_equivalent_dicts(self):
        a = {
            "x": np.array([1.0, 2.0]),
            "y": [1, 2, {"z": "sand"}],
        }
        b = {
            "x": np.array([1.0, 2.0]),
            "y": [1, 2, {"z": "sand"}],
        }

        self.assertTrue(f.deep_object_equivalent(a, b))

        b["y"][2]["z"] = "clay"
        self.assertFalse(f.deep_object_equivalent(a, b))
    
        
if __name__ == "__main__":
    unittest.main()
