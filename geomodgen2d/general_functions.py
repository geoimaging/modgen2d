# This file is part of geomodgen3d a Python package for ...
# Copyright (C) XXXX Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
# LICENSE

## Note: Added by Sanish (Feb 24, 2025)

import numpy as np
import scipy, re, warnings

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

def check_obstruction_id(obstruction_id):
    if not is_integer_value(obstruction_id):
        raise ValueError(f"Invalid obstruction_id, must be positive integer. Provided {obstruction_id}")
    obstruction_id = check_integer(float(obstruction_id))
    if obstruction_id<=0:
        raise ValueError(f"Invalid obstruction_id, must be positive integer. Provided {obstruction_id}")
            
    return obstruction_id

def is_valid_feature_id(added_prefix):
    """
    Checks if a prefix is valid based on rules:
    1. Must be a string.
    2. Must not contain digits or underscores.
    3. Must be <= 8 characters.
    4. Must not be empty or 'def'.
    """
    
    msg = f"Feature Ids/added_prefix must be a string, cannot have '_' or numbers. Cannot be '', or more than 8 lettered. Provided '{added_prefix}'"
    if not isinstance(added_prefix, str):
        return False, msg
    
    if added_prefix == "":
        return False, msg
    
    if added_prefix == "def":
        return False, "Feature id: 'def' is reserved for soils. It cannot be used for added_prefix."
    
    if len(added_prefix) > 8:
        return False, msg
    
    if re.search(r"[_0-9]", added_prefix):
        return False, msg
    
    return True, msg

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

def format_lith_ids(feature_id, vals):
    if feature_id == 'def':
        return [str(v) for v in vals]
    else:
        return [f"{feature_id}_{v}" for v in vals]
    
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
    
class FixedKeyDict(dict):
    """
    Dictionary with fixed keys: existing keys can be updated, 
    but new keys cannot be added. Works recursively for nested dicts.
    """
    def __setitem__(self, key, value):
        if key not in self:
            raise KeyError(f"Key '{key}' cannot be added to this dictionary.")
        
        current_value = self[key]
        # If both current value and new value are dicts, recursively wrap
        if isinstance(current_value, dict) and isinstance(value, dict):
            value = FixedKeyDict(value)
        
        # If type mismatch between dict and non-dict, raise error
        elif isinstance(current_value, dict) != isinstance(value, dict):
            raise TypeError(
                f"Cannot update key '{key}': type mismatch "
                f"(current type: {type(current_value).__name__}, "
                f"new type: {type(value).__name__})"
            )
        
        super().__setitem__(key, value)

def validate_processed_property_dict(processed_property_dict):
    """
    Validates if the given dictionary follows the required format:
    
    {
        'layer_ID': {'wet': {'mean': float, 'mean_bm': float (optional), 'stdev_or_cov': float, 'stdev_type':string}, 
                     'dry': {'mean': float, 'mean_bm': float (optional), 'stdev_or_cov': float, 'stdev_type':string}},
        'layer_ID2': {'wet': {'mean': float, 'mean_bm': float (optional), 'stdev_or_cov': float, 'stdev_type':string}, 
                     'dry': {'mean': float, 'mean_bm': float (optional), 'stdev_or_cov': float, 'stdev_type':string}},
        'layer_ID3': {'both': {'mean': float, 'mean_bm': float (optional), 'stdev_or_cov': float, 'stdev_type':string}},                      
        ...
    }

    Parameters:
    property_dict : dict
        Dictionary containing layer IDs as keys and their corresponding mean and standard deviation values as nested dictionaries.

    Returns:
    processed_property dict; adjusted if mean_bm (optional) is not provided.
    """
    if processed_property_dict is not None:
        assert isinstance(processed_property_dict, dict), "Input must be a dictionary."
        
        for key, value in processed_property_dict.items():
            assert isinstance(value, dict), f"Value for key '{key}' must be a dictionary."
            
            # Assert that layer_ID contains either 'wet' and 'dry' or 'both'
            assert ('wet' in value and 'dry' in value) or 'both' in value, f"Layer '{key}' must contain both 'wet' and 'dry', or 'both'."

            # Check for wet, dry, or all layer types
            for subkey, subvalue in value.items():
                if subkey!= 'layer0_air':
                    assert subkey in ['wet', 'dry', 'both'], f"Subkey '{subkey}' for key '{key}' is invalid. Expected 'wet', 'dry', or 'both'."
                    assert isinstance(subvalue, dict), f"Subvalue for '{subkey}' in layer '{key}' must be a dictionary."
                    
                    # Validate mean and stdev for each layer type
                    assert 'mean' in subvalue, f"'{subkey}' layer for key '{key}' must contain 'mean'."
                    assert 'stdev_or_cov' in subvalue, f"'{subkey}' layer for key '{key}' must contain 'stdev_or_cov'."
                    assert 'stdev_type' in subvalue, f"'{subkey}' layer for key '{key}' must contain 'stdev_type'."
                    assert isinstance(subvalue['mean'], (int, float)), f"'mean' for '{subkey}' in layer '{key}' must be a number."
                    assert isinstance(subvalue['stdev_or_cov'], (int, float)), f"'stdev_or_cov' for '{subkey}' in layer '{key}' must be a number."
                    assert subvalue['stdev_type'] in ['stdev', 'cov'], "stdev_type must be either 'stdev', or 'cov'."
                    
                    # 'mean_bm' is optional, but if present, must be a number
                    if 'mean_bm' in subvalue:
                        assert isinstance(subvalue['mean_bm'], (int, float)), f"'mean_bm' for '{subkey}' in layer '{key}' must be a number (if provided)."
                    else:
                        processed_property_dict[key][subkey]['mean_bm'] = 0.0

    return processed_property_dict

def deep_object_equivalent(a, b, rtol=1e-9, atol=1e-12, type_check=False):
    if a is b:
        return True
    if a is None or b is None:
        return a is b
    if type_check:
        if type(a) is not type(b):
            warnings.warn(f"Mismatch in Types:{type(a)} != {type(b)}")
            return False

    # Scalars
    if isinstance(a, (int, float, bool, str, np.number)):
        check = abs(a - b) <= (atol + rtol * abs(b)) if isinstance(a, float) else a == b
        if not check:
            warnings.warn(f"Mismatch in scalar number values: {a} != {b}")
        return check

    # numpy arrays
    if isinstance(a, np.ndarray):
        if a.shape != b.shape:
            warnings.warn(f"Mismatch in array shape:{a.shape} != {b.shape}")
            return False
        
        # --- Numeric arrays → allclose ---
        if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
            check = np.allclose(a, b, rtol=rtol, atol=atol)
            if not check:
                warnings.warn("Numeric array values do not match")
            return check

        # --- Non-numeric arrays (strings, objects) → exact match ---
        check = np.array_equal(a, b)
        if not check:
            warnings.warn("Non-numeric array values do not match")
        return check

    # lists / tuples
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            warnings.warn(f"Mismatch in list/tuple length:{len(a)} != {len(b)}")
            return False
        check = all(deep_object_equivalent(x, y, rtol, atol) for x, y in zip(a, b))
        if not check:
            warnings.warn(f"List/Tuples values does not match")
        return check

    # dict
    if isinstance(a, dict):
        if a.keys() != b.keys():
            warnings.warn(f"Mismatch in dictionary keys:{a.keys()} != {b.keys()}")
            return False
        check = all(deep_object_equivalent(a[k], b[k], rtol, atol) for k in a)
        if not check:
            warnings.warn(f"Dictionaries does not match")
        return check

    # objects: inspect real stored fields
    fields = set()
    if hasattr(a, "__dict__"):
        fields |= set(a.__dict__.keys())
    if hasattr(a, "__slots__"):
        fields |= set(a.__slots__)

    for f in fields:
        if not deep_object_equivalent(getattr(a, f, None), getattr(b, f, None), rtol, atol):
            warnings.warn(f"Mismatch in : {f}: {getattr(a, f, None)} != {getattr(b, f, None)}")
            return False

    return True

