# This file is part of <PROJECT>, a Python package for <DESCRIPTION>.
# Copyright (C) <YEAR> Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
# <LICENSE>

"""Testing tools."""

import unittest
import pathlib
import warnings

import numpy as np


def get_full_path(path, result_as_string=True):
    full_path = pathlib.Path(path).resolve().parent
    return str(full_path) if result_as_string else full_path


class TestCase(unittest.TestCase):

    def assertListAlmostEqual(self, list1, list2, **kwargs):
        if len(list1) != len(list2):
            raise AssertionError(f"List length mismatch: {len(list1)} != {len(list2)}")
    
        for a, b in zip(list1, list2):
            self.assertAlmostEqual(a, b, **kwargs)

    def assertNestedListEqual(self, list1, list2, **kwargs):
        if len(list1) != len(list2):
            msg = f"\nExpected:\n{list1}\nReturned:\n{list2})"
            raise AssertionError(msg)

        for l1, l2 in zip(list1, list2):
            self.assertListEqual(l1, l2, **kwargs)

    def assertArrayEqual(self, array1, array2):
        try:
            self.assertTrue(np.equal(array1, array2, casting='no').all())
        except AssertionError as e:
            msg = f"\nExpected:\n{array1}\nReturned:\n{array2})"
            raise AssertionError(msg) from e

    def assertArrayAlmostEqual(self, array1, array2, **kwargs):
        if kwargs.get("places", False):
            kwargs["atol"] = 1/(10**kwargs["places"])
            del kwargs["places"]

        if kwargs.get("delta", False):
            kwargs["atol"] = kwargs["delta"]
            del kwargs["delta"]

        try:
            self.assertTrue(np.allclose(array1, array2, **kwargs))
        except AssertionError as e:
            msg = f"\nExpected:\n{array1}\nReturned:\n{array2})"
            raise AssertionError(msg) from e

    def assertDictDeepAlmostEqual(self, dict1, dict2, tol=None, msg=None):
        """
        Recursively assert that two dictionaries are equal in keys and values,
        with optional tolerance for numeric and array comparisons.
        
        Parameters
        ----------
        dict1, dict2 : dict
            Dictionaries to compare.
        tol : float, optional
            Absolute tolerance for approximate equality (applies to floats and arrays).
        msg : str, optional
            Custom message for AssertionError.
        """
        # Check key sets
        if set(dict1.keys()) != set(dict2.keys()):
            missing_1 = set(dict2.keys()) - set(dict1.keys())
            missing_2 = set(dict1.keys()) - set(dict2.keys())
            raise AssertionError(
                msg or f"Dictionary keys mismatch.\nMissing in dict1: {missing_1}\nMissing in dict2: {missing_2}"
            )

        for key in dict1:
            val1, val2 = dict1[key], dict2[key]

            # Nested dicts
            if isinstance(val1, dict) and isinstance(val2, dict):
                self.assertDictDeepAlmostEqual(val1, val2, tol=tol, msg=msg)

            # Numpy arrays
            elif isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
                if tol is not None:
                    self.assertArrayAlmostEqual(val1, val2, atol=tol)
                else:
                    self.assertArrayEqual(val1, val2)

            # Lists or tuples
            elif isinstance(val1, (list, tuple)) and isinstance(val2, (list, tuple)):
                if len(val1) != len(val2):
                    raise AssertionError(
                        msg or f"List length mismatch at key '{key}': {len(val1)} != {len(val2)}"
                    )
                for i, (a, b) in enumerate(zip(val1, val2)):
                    if isinstance(a, (dict, list, np.ndarray)):
                        self.assertDictDeepAlmostEqual(
                            {f"{key}[{i}]": a}, {f"{key}[{i}]": b}, tol=tol, msg=msg
                        )
                    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                        if tol is not None:
                            self.assertAlmostEqual(a, b, delta=tol, msg=msg)
                        else:
                            self.assertEqual(a, b, msg)
                    else:
                        self.assertEqual(a, b, msg)

            # Scalars (float, int, str, etc.)
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                if tol is not None:
                    self.assertAlmostEqual(val1, val2, delta=tol, msg=msg)
                else:
                    self.assertEqual(val1, val2, msg)
            else:
                self.assertEqual(val1, val2, msg or f"Value mismatch at key '{key}': {val1} != {val2}")

    def assertWarningContains(self, func, *args, contains="", **kwargs):
        """
        Run function `func(*args, **kwargs)` and assert that at least one warning contains `text`.
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            func(*args, **kwargs)

        # Must have at least one warning
        self.assertTrue(w, "No warnings were raised.")

        # Must contain the word in at least one warning message
        if not any(contains in str(wi.message) for wi in w):
            all_msgs = [str(wi.message) for wi in w]
            raise AssertionError(
                f"No warning contained the text '{contains}'. "
                f"Warnings were: {all_msgs}"
            )
