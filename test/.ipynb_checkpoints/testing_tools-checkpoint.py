# This file is part of <PROJECT>, a Python package for <DESCRIPTION>.
# Copyright (C) <YEAR> Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
# <LICENSE>

"""Testing tools."""

import unittest
import pathlib

import numpy as np


def get_full_path(path, result_as_string=True):
    full_path = pathlib.Path(path).resolve().parent
    return str(full_path) if result_as_string else full_path


class TestCase(unittest.TestCase):

    def assertListAlmostEqual(self, list1, list2, **kwargs):
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
