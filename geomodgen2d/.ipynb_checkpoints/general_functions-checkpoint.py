# This file is part of geomodgen3d a Python package for ...
# Copyright (C) XXXX Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
# LICENSE

## Note: Added by Sanish (Feb 24, 2025)

import numpy as np
import scipy

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
    else:
        a = int(a)
    return a

def is_integer_value(value):
    try:
        # Convert value to float, then to int, and back to string to check if integer-like
        return float(value).is_integer()
    except ValueError:
        return False

def remeshing_2D_matrix(x_old, x_new, z_old, z_new, matrix_2d, interp_method='linear'):
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
    """
    del_x_old = x_old[1] - x_old[0]
    del_z_old = z_old[1] - z_old[0]
    del_x_new = x_new[1] - x_new[0]
    del_z_new = z_new[1] - z_new[0]
    new_matrix_2d = matrix_2d
    if del_x_old != del_x_new or del_z_old!=del_z_new:
        interp = scipy.interpolate.RegularGridInterpolator((x_old,z_old), matrix_2d,
                                     bounds_error=False, fill_value=0, method=interp_method) #None means extrapolate outside the bounds
        xg, zg = np.meshgrid(x_new, z_new, indexing='ij')
        new_matrix_2d = interp((xg, zg))
    return new_matrix_2d