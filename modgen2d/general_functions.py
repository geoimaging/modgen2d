# This file is part of geomodgen3d a Python package for ...
# Copyright (C) XXXX Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
# LICENSE

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
        'layer_ID': {'wet': {'mean': float, 'mean_slope_with_depth': float (optional), 'stdev_or_cov': float, 'stdev_type':string}, 
                     'dry': {'mean': float, 'mean_slope_with_depth': float (optional), 'stdev_or_cov': float, 'stdev_type':string}},
        'layer_ID2': {'wet': {'mean': float, 'mean_slope_with_depth': float (optional), 'stdev_or_cov': float, 'stdev_type':string}, 
                     'dry': {'mean': float, 'mean_slope_with_depth': float (optional), 'stdev_or_cov': float, 'stdev_type':string}},
        'layer_ID3': {'both': {'mean': float, 'mean_slope_with_depth': float (optional), 'stdev_or_cov': float, 'stdev_type':string}},                      
        ...
    }

    Parameters:
    property_dict : dict
        Dictionary containing layer IDs as keys and their corresponding mean and standard deviation values as nested dictionaries.

    Returns:
    processed_property dict; adjusted if mean_slope_with_depth (optional) is not provided.
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
                    
                    # 'mean_slope_with_depth' is optional, but if present, must be a number
                    if 'mean_slope_with_depth' in subvalue:
                        assert isinstance(subvalue['mean_slope_with_depth'], (int, float)), f"'mean_slope_with_depth' for '{subkey}' in layer '{key}' must be a number (if provided)."
                    else:
                        processed_property_dict[key][subkey]['mean_slope_with_depth'] = 0.0

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