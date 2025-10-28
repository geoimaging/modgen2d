# This file is part of geomodgen3d a Python package for ...
# Copyright (C) XXXX Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
# LICENSE

## Note: Added by Sanish (Feb 24, 2025)

import numpy as np
import scipy, re
import geomodgen2d.general_functions as f
import geomodgen2d.utils_2d as utils_2d

def get_rotation_matrix(angles_in_degrees):
    """
    Computes a 3D rotation matrix given the Euler angles (gamma, beta, theta) in degrees.

    Parameters:
    angles_in_degrees : tuple or list
        A sequence of three angles (gamma, beta, theta) in degrees
        - theta: Rotation about the X-axis
        - beta: Rotation about the Y-axis
        - gamma: Rotation about the Z-axis

    Returns:
    numpy.ndarray: A 3x3 rotation matrix obtained by sequentially applying the three rotations.

    The transformation follows the standard intrinsic Tait-Bryan angles (Z-Y-X rotation order).
    """
    gamma, beta, theta = np.radians(angles_in_degrees)
    
    # Create the rotation matrix using the axis-angle formula
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    R_z_theta = np.array([
        [1, 0, 0],
        [0, cos_theta, -sin_theta],
        [0, sin_theta, cos_theta],
    ])
    
    cos_beta = np.cos(beta)
    sin_beta = np.sin(beta)
    R_y_beta = np.array([
        [cos_beta, 0, sin_beta],
        [0, 1, 0],
        [-sin_beta, 0, cos_beta],
    ])
    
    cos_gamma = np.cos(gamma)
    sin_gamma = np.sin(gamma)
    R_x_gamma = np.array([
        [cos_gamma, -sin_gamma, 0],
        [sin_gamma, cos_gamma, 0],
        [0, 0, 1],
    ])

    return np.matmul(R_z_theta, np.matmul(R_y_beta, R_x_gamma))

def check_rotation_angles(angles_in_degrees):
    if angles_in_degrees is None:
        return False
    assert isinstance(angles_in_degrees, (list, tuple)), "Input must be a list or tuple"
    assert len(angles_in_degrees) == 3, "Input must contain exactly three angles"
    assert all(isinstance(a, (int, float)) for a in angles_in_degrees), "All angles must be int or float"
    gamma_d, beta_d, theta_d = angles_in_degrees
    
    if gamma_d == 0 and beta_d == 0 and theta_d == 0:
        return False

    return True

def get_table_from_utils2d(Utils2d_class: utils_2d, shift_points2d):
    """
    Converts matrix format in Utils2d class to tabular format z|x|| grid_val

    Parameters:
    Utils2d_class : (Utils2D): 
        Utils2D class instance
    shift_points2d : (tuple or list)
        Coordinate (z, x) of the shift_point (reference point of the utils (in actual_coordinates) will be shifted to this point.)
    """
    
    grid = Utils2d_class.utilsgrid
    grid = np.pad(grid, pad_width=1, mode='constant', constant_values=0) #Note padded one so ref_coord also add 1 
    # Zero-padding in order to have zeroes in all outer cases during nearest interpolation later.

    ref_coord3d_1 = np.array([(Utils2d_class.ref_coord2d_in_utilsgrid[0] - 1 + 1) * Utils2d_class.del_z_utils, 0, (Utils2d_class.ref_coord2d_in_utilsgrid[1] - 1 + 1)*Utils2d_class.del_x_utils]) # [z,y,x] from [z,x]
    ref_coord3d_2 = np.array([(Utils2d_class.ref_coord2d_in_utilsgrid[0] - 1 + 1) * Utils2d_class.del_z_utils, 1, (Utils2d_class.ref_coord2d_in_utilsgrid[1] - 1 + 1)*Utils2d_class.del_x_utils]) # -1 because whole x_ranges is shifted -1 (below), and +1 because of padding
    
    # Create a 2D grid in [z, x] coordinates
    z, x = np.meshgrid(np.arange(grid.shape[0]), np.arange(grid.shape[1]), indexing='ij')
    # Flatten the arrays
    x = x.flatten()*Utils2d_class.del_x_utils-Utils2d_class.del_x_utils
    z = z.flatten()*Utils2d_class.del_z_utils-Utils2d_class.del_z_utils
    grid_val = grid.flatten()
    # Format of points = |z|y|x|grid_val|
    
    points = [[ref_coord3d_1[0], ref_coord3d_1[1], ref_coord3d_1[2]],  #First point is reference point
                [ref_coord3d_2[0], ref_coord3d_2[1], ref_coord3d_2[2]],  #Purpose of the second point is to compute the direction cosine of the extrusion axis.
                ] 
    #points = [ref_coord3d_1.tolist(), ref_coord3d_2.tolist()]
    grid2d = [-1000, -1000] # Special values for reference points

    # Remaining points are points to represent utils and its id.
    for i,k,val in zip(x,z,grid_val):
        assert f.is_integer_value(val)
        points.append([k,ref_coord3d_1[1],i])
        grid2d.append(int(val))
        
    shift_points3d_in_grids = [shift_points2d[0], 0, shift_points2d[1]]
    points=np.array(points) + np.array(shift_points3d_in_grids)
    
    points2d = points.T  #Note: Here it only contains cross-sectional points
    grid2d = np.array(grid2d)

    return points2d, grid2d

def get_direction_cosines_from_ref_points(reference_pointA, reference_pointB):
    """
    Computes the direction cosines and angles between two points in a (z, y, x) coordinate system.

    Parameters:
    reference_pointA : (tuple or list): 
        Coordinates (z1, y1, x1) of the first point.
    reference_pointB : (tuple or list)
        Coordinates (z2, y2, x2) of the second point.
    """
    
    z1, y1, x1 = reference_pointA
    z2, y2, x2 = reference_pointB

    d = np.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
    l = [(x2-x1)/d, (y2-y1)/d, (z2-z1)/d]
    return l, [np.degrees(np.arccos(i)) for i in l]

def get_y_projection(y, all_z0_y0_x0, common_angle):
    """
    Projects a point in the y-direction based on a common angle and initial coordinates.

    Args:
        y (float): The y-coordinate for the projection.
        all_z0_y0_x0 (np.ndarray): An array of initial coordinates in the order [z0, y0, x0].
        common_angle (tuple): A tuple of angles (alpha i.e. w.r.t x, beta i.e. w.r.t y, gamma i.e. w.r.t z) in degrees, representing the orientation.

    Returns:
        tuple: A tuple (z, y_t, x) representing the projected coordinates in the x, y, and z directions.
    """

    all_x0s = all_z0_y0_x0[2]
    all_y0s = all_z0_y0_x0[1]
    all_z0s = all_z0_y0_x0[0]
    # Angles with axes in degrees (alpha, beta, gamma)
    alpha, beta, gamma = common_angle
    
    # Convert angles to radians
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    gamma = np.radians(gamma)
    
    # Compute direction cosines
    l = np.cos(alpha)
    m = np.cos(beta)
    n = np.cos(gamma)
    # print(l,m,n)
    
    assert np.round(l**2+m**2+n**2, 4) == 1, f"For valid angles of 3D line with respect to axis, l^2 + m^2 + n^2 cannot be one. Provided angles {common_angle} lead to sum of squares as {np.round(l**2+m**2+n**2, 4)}"
    assert np.round(m,4)!=0, "Error: Common angle about y-axis cannot be 90 degree (Numerical Error)."
    
    t_y = (np.ones_like(all_x0s)*y - all_y0s)/m
    x = all_x0s + t_y * l
    y_t = all_y0s + t_y * m
    z = all_z0s + t_y * n

    return z,y_t,x

def get_rotated_utils_grid(utils2d_class, shift_points2d, rotation_angle_in_degrees, allow_z_axis_rotation_only=True):
    """
        Rotates the grid in 3D space using a rotation matrix.

        Parameters:
        rotation_angle_in_degrees (list of float): 
            Rotation angles [gamma, beta, theta] in degrees.
    """
    points2d, grid2d = get_table_from_utils2d(utils2d_class, shift_points2d)
    angle_description = f'{rotation_angle_in_degrees[0]} about x-axis, {rotation_angle_in_degrees[1]} about y-axis, and {rotation_angle_in_degrees[2]} about z-axis'
    if allow_z_axis_rotation_only:
        assert rotation_angle_in_degrees[0] == 0 and rotation_angle_in_degrees[1] == 0, "Only z-axis rotation is allowed. So angles_in_degree must be [0,0,##]"
        angle_description = f'{rotation_angle_in_degrees[2]} about z-axis'
        
    R_matrix = get_rotation_matrix(rotation_angle_in_degrees)
    center = points2d[:, [0]]
    rotated_point = np.matmul(R_matrix, (points2d-center))+center
    points2d = rotated_point
    _, angles = get_direction_cosines_from_ref_points(points2d[:, 0], points2d[:, 1]) 
    description_add = f', then rotated by angle (in degrees) of {angle_description}'

    z,y_t,x = get_y_projection(0, points2d[:, 2:], angles)
    layered_matrix_coord_table = np.array([z, x])
    layered_matrix_grid_table = np.array(grid2d[2:])
    
    return layered_matrix_coord_table, layered_matrix_grid_table, description_add


