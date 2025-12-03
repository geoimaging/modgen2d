import numpy as np
from pathlib import Path
from testing_tools import TestCase, unittest, get_full_path
# Assuming the original TestCase class is in a module named `testing_tools`
# from testing_tools import TestCase, get_full_path  

class TestTestCaseAssertions(unittest.TestCase):

    def setUp(self):
        # Create an instance of your custom TestCase to use its assertions
        self.tc = TestCase(methodName='runTest')

    def test_get_full_path(self):
        path = "."
        full_path_str = get_full_path(path)
        full_path_obj = get_full_path(path, result_as_string=False)
        self.assertTrue(isinstance(full_path_str, str))
        self.assertTrue(isinstance(full_path_obj, Path))
        self.assertTrue(full_path_str.endswith(str(full_path_obj.name)))

    def test_assertListAlmostEqual(self):
        list1 = [1.0, 2.0, 3.000001]
        list2 = [1.0, 2.0, 3.0]
        self.tc.assertListAlmostEqual(list1, list2, places=5) #round(a-b,5)
        
        # Test failure
        with self.assertRaises(AssertionError):
            self.tc.assertListAlmostEqual(list1, list2, places=6)
            
        with self.assertRaises(AssertionError):
            self.tc.assertListAlmostEqual([1, 2, 3, 4], list2, places=5)
            
        self.tc.assertListAlmostEqual([1, 2, 3, 'a'], [1,2,3,'a'])

    def test_assertNestedListEqual(self):
        list1 = [[1, 2], [3, 4]]
        list2 = [[1, 2], [3, 4]]
        self.tc.assertNestedListEqual(list1, list2)
        # Test failure
        with self.assertRaises(AssertionError):
            self.tc.assertNestedListEqual([[[1]]], [[1]])
            
        with self.assertRaises(AssertionError):
            self.tc.assertNestedListEqual([[1,2]], [[1,2,3]])

        with self.assertRaises(TypeError):
            self.tc.assertNestedListEqual([[1]], 1)
            
        with self.assertRaises(TypeError):
            self.tc.assertNestedListEqual(1, 1)

        with self.assertRaises(AssertionError):
            self.tc.assertNestedListEqual([[1,2], [2]], [[1,2], [2,1]])
        
        self.tc.assertNestedListEqual([[1,2], [2]], [[1,2], [2]])
        

    def test_assertArrayEqual(self):
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([1, 2, 3])
        self.tc.assertArrayEqual(arr1, arr2)
        with self.assertRaises(AssertionError):
            self.tc.assertArrayEqual(arr1, np.array([1, 2, 4]))
            
        with self.assertRaises(ValueError):
            self.tc.assertArrayEqual(arr1, np.array([1, 2, 3, 4]))
            
        self.tc.assertArrayEqual([[1, 2, 3, 4]], np.array([[1, 2, 3, 4]]))
        
        with self.assertRaises(ValueError):
            self.tc.assertArrayEqual([[1, 2, 3, 4], [1,2]], np.array([[1, 2, 3, 4], [1,2]]))
            
        with self.assertRaises(AssertionError):
            self.tc.assertArrayEqual([[np.nan, 2, 3, 4]], [[np.nan, 2, 3, 4]])
            
        with self.assertRaises(np.core._exceptions._UFuncInputCastingError):
            self.tc.assertArrayEqual([[1, 2.0, 3., 4]], np.array([[1, 2, 3, 4]]))
        
        self.tc.assertArrayEqual([1, 2, 3, 'a'], [1,2,3,'a'])
        

    def test_assertArrayAlmostEqual(self):
        arr1 = np.array([1.0, 2.0, 3.000001])
        arr2 = np.array([1.0, 2.0, 3.0])
        self.tc.assertArrayAlmostEqual(arr1, arr2, atol=1e-5)
        self.tc.assertArrayAlmostEqual([[1, 2.0, 3., 4]], np.array([[1, 2, 3, 4]]))
        

    def test_assertDictDeepAlmostEqual_scalars(self):
        d1 = {"a": 1, "b": 2.00001}
        d2 = {"a": 1, "b": 2.0}
        self.tc.assertDictDeepAlmostEqual(d1, d2, tol=1e-4)

    def test_assertDictDeepAlmostEqual_nested(self):
        d1 = {"x": {"y": [1.0, 2.0]}, "z": np.array([1.0, 2.0])}
        d2 = {"x": {"y": [1.0, 2.0]}, "z": np.array([1.0, 2.0])}
        self.tc.assertDictDeepAlmostEqual(d1, d2)

    def test_assertDictDeepAlmostEqual_lists_and_arrays(self):
        d1 = {"list": [1.0, 2.0], "arr": np.array([3.0, 4.0])}
        d2 = {"list": [1.0, 2.0], "arr": np.array([3.0, 4.0])}
        self.tc.assertDictDeepAlmostEqual(d1, d2)

    def test_assertDictDeepAlmostEqual_failures(self):
        # Key mismatch
        d1 = {"a": 1}
        d2 = {"b": 1}
        with self.assertRaises(AssertionError):
            self.tc.assertDictDeepAlmostEqual(d1, d2)

        # Value mismatch
        d1 = {"a": 1}
        d2 = {"a": 2}
        with self.assertRaises(AssertionError):
            self.tc.assertDictDeepAlmostEqual(d1, d2)

        # List mismatch
        d1 = {"l": [1, 2]}
        d2 = {"l": [1, 3]}
        with self.assertRaises(AssertionError):
            self.tc.assertDictDeepAlmostEqual(d1, d2)

        # Dict mismatch
        d1 = {"l": {2: [1, 3]}}
        d2 = {"l": [1, 3]}
        with self.assertRaises(AssertionError):
            self.tc.assertDictDeepAlmostEqual(d1, d2)
            
        # Array mismatch
        d1 = {"a": np.array([1, 2])}
        d2 = {"a": np.array([1, 3])}
        with self.assertRaises(AssertionError):
            self.tc.assertDictDeepAlmostEqual(d1, d2)
            
        d1 = {"a": np.array([1, 2])}
        d2 = {"a": np.array([1, 2, 3])}
        with self.assertRaises(ValueError):
            self.tc.assertDictDeepAlmostEqual(d1, d2)

if __name__ == "__main__":
    unittest.main()
