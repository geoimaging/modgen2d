# This file is part of geomodgen3d a Python package for ...
# Copyright (C) XXXX Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
# LICENSE

## Note: Added by Sanish (Feb 24, 2025)

import numpy as np
import scipy, re

def is_divisible(a, b, tol=1e-9):
    if a<b: raise ValueError("ERROR: a cannot be less than b")
    if a<=0: raise ValueError(f"ERROR: a={a} must be positive")
    if b<0: raise ValueError(f"ERROR: b={b} must be positive")
    
    if b == 0:
        return False  # Division by zero is undefined
    remainder = a % b
    return abs(remainder) < tol or abs(remainder - b) < tol

def check_integer(a):
    if not is_divisible(abs(a), 1):
        raise ValueError(f"Shift values must be integers. Provided {a}" )
    return int(round(a))

def is_integer_value(value):
    try: 
        # Convert value to float, then to int, and back to string to check if integer-like
        return float(value).is_integer()
    except ValueError:
        return False
        
def is_close(a, b, tol=1e-8):
    return abs(a - b) < tol
    
def get_span_del_from_ranges(ranges):
    return (ranges[0] + ranges[-1]), (ranges[1] - ranges[0])

def check_obstruction_id(obstruction_id):
    if not is_integer_value(obstruction_id):
        raise ValueError(f"Invalid obstruction_id, must be positive integer. Provided {obstruction_id}")
    obstruction_id = check_integer(float(obstruction_id))
    if obstruction_id<=0:
        raise ValueError(f"Invalid obstruction_id, must be positive integer. Provided {obstruction_id}")
            
    return obstruction_id

def is_valid_prefix(added_prefix):
    case_ = True
    if added_prefix is not None:
        case_ = added_prefix != "" and len(added_prefix) <= 8 and not bool(re.search(r"[_0-9]", added_prefix))
    return case_  

def coordinate_vars(x_ranges, z_ranges):
    del_x, del_z = x_ranges[0]*2, z_ranges[0]*2
    span_x, span_z = x_ranges[-1]+del_x/2, z_ranges[-1]+del_z/2
    return span_x, span_z, del_x, del_z

def format_value(val):
    return f'{int(float(val))}' if is_integer_value(val) else f'{val}'

def remeshing_2D_matrix(x_old, x_new, z_old, z_new, matrix_2d, interp_method='linear', extrapolate=True):
    """
    Refines/coarsens a 2D matrix by interpolating values based on new grid coordinates.

    Arguments:
    x_old : array-like
        The original x coordinates.
    x_new : array-like
        The new x coordinates.
    z_old : array-like
        The original z coordinates.
    z_new : array-like
        The new z coordinates.
    matrix_2d : 2D array
        The original 2D matrix to be interpolated.
    interp_method : str, optional, default 'linear'
        The interpolation method ('linear', 'nearest', 'cubic', etc.).
    extrapolate : boolean, optional
        Flag if extrapolate
    """
    new_matrix_2d = matrix_2d
    
    if extrapolate:
        fill_value = None
    else:
        fill_value = 0
        
    if not np.array_equal(x_old, x_new) or not np.array_equal(z_old, z_new):
        interp = scipy.interpolate.RegularGridInterpolator((x_old,z_old), matrix_2d,
                                     bounds_error=False, fill_value=fill_value, method=interp_method) #None means extrapolate outside the bounds
        xg, zg = np.meshgrid(x_new, z_new, indexing='ij')
        new_matrix_2d = interp((xg, zg))
    return new_matrix_2d

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
    
class ReadOnlyDict(dict):
    def __setitem__(self, key, value):
        if key not in self:
            raise KeyError(f"Key '{key}' cannot be added to this dictionary.")
        else:
            super().__setitem__(key, value)