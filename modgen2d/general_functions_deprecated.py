# This file is part of geomodgen3d a Python package for ...
# Copyright (C) XXXX Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
# LICENSE

## Note: Added by Sanish (Feb 23, 2026)
## Functions deprecated from general_functions. Will be removed once confirmed unused.

import numpy as np
import scipy, re, warnings

def safe_equal(a, b, tol=1e-10):
    # both None
    if a is None and b is None:
        return True
    # one None
    if (a is None) ^ (b is None):
        return False

    # numpy arrays
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.shape != b.shape:
            return False
        return np.allclose(a, b, atol=tol, rtol=tol)

    # scalars
    return a == b
    
def get_span_del_from_ranges(ranges):
    return (ranges[0] + ranges[-1]), (ranges[1] - ranges[0])

def validate_feature_ids_list(features_ids_list:list):
    """
    Validates a list of feature IDs.
    
    Rules:
    1. Must contain 'def'.
    2. All other keys must be valid feature IDs (checked via is_valid_feature_id).
    """
    if 'def' not in features_ids_list:
        raise KeyError("Key must contain 'def'")
    
    invalid = [
        k for k in features_ids_list
        if k != 'def' and not is_valid_feature_id(k) # only one underscore allowed at end
    ]
    if invalid:
        raise KeyError(f"Invalid keys (must end with '_', no digits, no other underscores): {invalid}")
    
def coordinate_vars(x_ranges, z_ranges):
    del_x, del_z = x_ranges[0]*2, z_ranges[0]*2
    span_x, span_z = x_ranges[-1]+del_x/2, z_ranges[-1]+del_z/2
    return span_x, span_z, del_x, del_z

def upsample_2d_blocks(x_coord, y_coord, x_coord_new, y_coord_new, matrix_2d):
    """
    Upsamples a 2D matrix by repeating elements block-wise.

    Arguments:
    x_coord : array-like
        Original x coordinates.
    y_coord : array-like
        Original y coordinates.
    x_coord_new : array-like
        New x coordinates (must be finer and evenly divisible).
    y_coord_new : array-like
        New y coordinates.
    matrix_2d : 2D array
        Original 2D data.

    Returns:
    upsampled_matrix : 2D array
        The upsampled 2D matrix.
    """
    span_x, del_x = get_span_del_from_ranges(x_coord)
    span_xn, del_xn = get_span_del_from_ranges(x_coord_new)
    span_y, del_y = get_span_del_from_ranges(y_coord)
    span_yn, del_yn = get_span_del_from_ranges(y_coord_new)

    assert is_close(span_x, span_xn), "x_coord and x_coord_new must span the same range"
    assert is_close(span_y, span_yn), "y_coord and y_coord_new must span the same range"
    assert del_x >= del_xn and del_y >= del_yn, "New grid must be finer (upsampling only)"

    n_x = check_integer(del_x / del_xn)
    n_y = check_integer(del_y / del_yn)

    return np.kron(matrix_2d, np.ones((n_y, n_x), dtype=matrix_2d.dtype))

def get_nearest_centered_grid_point(val: float, grid_spacing: float, return_index:bool):
    """
    Find the nearest grid point to a given value assuming the grid points are centered at (grid_spacing / 2).
    
    For example, if grid spacing is 0.2, the grid points are located at:
        0.1, 0.3, 0.5, 0.7, ...
    
    This function returns the nearest such point to the input `val`.

    Parameters:
    val : float
        The target value to find the nearest grid point for.
    grid_spacing : float
        The spacing between grid points.
    return_index : bool, optional
        If True, returns the grid index `n` instead of the grid point. Default is False.
    """
    n = int(round((val - grid_spacing / 2) / grid_spacing))
    if return_index:
        return n
    return get_centered_grid_point_from_index(n, grid_spacing)

def get_centered_grid_point_from_index(n: int, grid_spacing: float):
    """
    Give centered_grid_point, based on the index.
    
    Parameters:
    n : int
        grid index
    grid_spacing : float
        The spacing between grid points.
    """
    return grid_spacing / 2 + n * grid_spacing

def get_nearest_ref_point_in_grid_from_utilsgrid(utils2D_instance, del_x, del_z):
    # Make sure utilsgrid is list of integers of size 2.
    refining_factor = utils2D_instance.refining_factor

    assert is_close(del_x, refining_factor * utils2D_instance.del_x_utils), (
        f"Inconsistent 'del_x' value detected: Expected 'del_x' = refining_factor ({refining_factor}) × del_x_utils ({utils2D_instance.del_x_utils}) = {refining_factor * utils2D_instance.del_x_utils}, but received {del_x}.")

    assert is_close(del_z, refining_factor * utils2D_instance.del_z_utils), (
        f"Inconsistent 'del_z' value detected: Expected 'del_z' = refining_factor ({refining_factor}) × del_z_utils ({utils2D_instance.del_z_utils}) = {refining_factor * utils2D_instance.del_z_utils}, but received {del_z}.")

    ref_coord2d_in_utilsgrid = utils2D_instance.ref_coord2d_in_utilsgrid
    ref_coord2d = [ref_coord2d_in_utilsgrid[0]*utils2D_instance.del_z_utils, ref_coord2d_in_utilsgrid[1]*utils2D_instance.del_x_utils]
    ref_coord2d_in_grid = [int(round(ref_coord2d[0]/del_z,0)), int(round(ref_coord2d[1]/del_x,0))]
    return ref_coord2d_in_grid

def replace_vals_in_array(np_array, replace_list = None, replace_with = None):
    """
    Replaces values in a numpy array based on a provided list.

    Arguments:
    np_array : 2D or 3D numpy array
        The array in which values are to be replaced.
    replace_list : list, optional
        The list of values to be replaced. If None, replaces all unique values in the array.
    replace_with : list, optional
        The list of values to replace with. If None, only returns the indices.

    Returns:
    2D or 3D numpy array or indices matrix
        The modified numpy array with replaced values or the indices where replacements occur.
    """
    if replace_list is None:
        replace_list = np.arange(len(np.unique(np_array)))
        
    assert np.isin(np_array, replace_list).all(), f"all elements of np_array must be in the replace_list. Not all of {np.unique(np_array)} is in {replace_list}"
    assert len(replace_list) == len(set(replace_list)), "all elements of replace_list must be unique"
    indices_matrix = np.searchsorted(replace_list, np_array)

    if replace_with is None:
        return indices_matrix
    else:
        assert np.issubdtype(indices_matrix.dtype, np.integer), "np_array must be integers"
        # replaced_mat = replace_with[indices_matrix]

        replace_map = {value: replace_with[idx] for idx, value in enumerate(replace_list)}
        replaced_mat = np.vectorize(replace_map.get)(np_array)

        # replaced_mat = np.take(replace_with, indices_matrix.flatten())
        return replaced_mat