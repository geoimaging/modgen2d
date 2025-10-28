# This file is part of geomodgen2D a Python package for ...
# Copyright (C) XXXX Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
# LICENSE

"""Define a three-dimensional domain that defines lithology."""

import numpy as np
import matplotlib.pyplot as plt
import geomodgen2d.general_functions as f
import geomodgen2d.utils_2d as utils_2d
import geomodgen2d.utils_3d_functions as utils_3d_f
import copy,scipy
import matplotlib.colors as mcolors

from geomodgen2d.domain2d import Domain2D, check_for_remeshing_coordinate_compatibility

        # self.lithology = np.array(lithology).astype(int)
        # self.lithology_map = dict(lithology_map)

# Lithological from boundary
# Lithological from utils

class LithologicalDomain2DFunctions(Domain2D):
    """
    Class representing a 2D matrix (with layer_ID) with layers that can be created from a boundary or utility class.
    """
    def __init__(self, span_x: float, span_z: float, del_x: float, del_z: float, name: str):
        """
        Initializes the LithogolicalDomain2D instance with given spatial limits, and spacing.
        
        Parameters:
        span_x, span_z : float
            The upper limit for the x, and z-coordinate range.
        del_x, del_z : float
            The spacing interval for x, and z-coordinates.
        name: str
            The name of lithologicaldomain
        """
        super().__init__(span_x, span_z, del_x, del_z, name)
        self.check = False
        self.gwt_depth=None
        self.lm_type = 'NA'
        self.added_prefix = False
        self.boundary_class = None
        self.read_only=False
        self.layered_matrix = None
        self.n_layers = None
        self.overlap = False
        self.utils_description = 'No utils'
        self.surface_boundary = None
        
    def print(self):
        print(f"N_z_coord = {self.layered_matrix.shape[0]}, N_x_coord = {self.layered_matrix.shape[1]}")
        print("Layered Matrix : \n", self.layered_matrix) 
        
    def plot2d(self, ax=None, discrete_point_size=0, legend=True,
        color_map = {
        'def': plt.get_cmap('tab20', 10),      # For integer values
        'U_': plt.get_cmap('Set3', 10)   # For "U-{x}" values
    }):
        """
        Plots a 2D section of the layered matrix.

        Parameters:
            ax: The matplotlib axes object for the plot (default is None, which creates a new figure).
            idx: A list specifying the axis and index to slice (e.g., ['x', 0]).
            color_map: A dictionary that defines the color map for the values in the matrix.
        """
        if ax is None:
            fig,ax = plt.subplots()

        unique_values = np.unique(self.layered_matrix)
        color_mapping = {} 
        for value in unique_values:
            assigned = False
            for prefix, cmap in color_map.items():
                if value == 'X':
                    color_mapping[value] = (1.,1.,1.,1.)#'#ffffff'
                    assigned = True
                    
                elif prefix == 'def' and f.is_integer_value(value):
                    if value == 0 or value == '0':
                        color_mapping[value] = '#ffffff'
                    else:
                        # If no prefix, assume integer or digit
                        index = int(float(value)) % 10
                        color_mapping[value] = cmap(index)
                    assigned = True
                    break
                elif isinstance(value, str) and value.startswith(prefix):
                    # For prefixed values
                    index = int(float(value[len(prefix):])) % 10
                    color_mapping[value] = cmap(index)
                    assigned = True
                    break
            
            # If no pattern matched, assign a random color
            if not assigned:
                color_mapping[value] = "#" + ''.join([np.random.choice(list('0123456789ABCDEF')) for _ in range(6)])

        int_map = {value: idx for idx, value in enumerate(unique_values)}
        integer_mapped_array = np.vectorize(int_map.get)(self.layered_matrix)
    
        # Create a colormap from the color mapping
        colors = [color_mapping[value] for value in unique_values]
        fixed_cmap = mcolors.ListedColormap(colors)
        
        if self.dim == 1:
            x_ranges_plt = [-self.span_z/10, self.span_z/10]
            integer_mapped_array = np.ones([len(integer_mapped_array), 2])*integer_mapped_array
        else:
            x_ranges_plt = [0, self.span_x]
            
        # Plot the data using imshow with the fixed colormap
        cax = ax.imshow(integer_mapped_array, cmap=fixed_cmap, extent=[x_ranges_plt[0], x_ranges_plt[1], self.span_z, 0], interpolation='none')
        
        z_data, x_data = np.meshgrid(self.z_ranges, self.x_ranges, indexing='ij')
        if discrete_point_size!=0:
            ax.scatter(x_data.flatten(), z_data.flatten(), c = [color_mapping[value] for value in self.layered_matrix.flatten()], edgecolors='k', s=discrete_point_size)
            
        ax.set(xlim=x_ranges_plt, ylim = [self.span_z, 0])
        # Create a custom legend
        if legend:
            handles = [plt.Line2D([0], [0], marker='s', color=color_mapping[value], markersize=10, linestyle='') for value in unique_values]
            ax.legend(handles, unique_values, title="Legend", bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.axis('scaled')
        ax.set(
            xlabel='X',
            ylabel='Z',
        )
        return ax

    def merge_with_another_lithological_domain(self, lithologicalDomainOther, lit_id_none_list:list = ['X']):
        """
        Merges two lithological_domains.

        Parameters:
            self, lithologicalDomainOther: Two lithologicalDomains to merge. 
            lit_id_none_list (list of str): List of lithological_id(s) that represent 'none' (as in utils lithological domain).

        Note: Priority given to Other case. i.e, the lit_value of Other will replace the one in self unless the Other case has a value from lit_id_none_list.
          
        Internally, updates the current layered_matrix after merging.
        """
        if self.lm_type == "NA":
            ## If NA then just copy the other lithological domain as itself.
            self.__dict__.update(lithologicalDomainOther.__dict__)
        else:
            assert lithologicalDomainOther.lm_type!="NA", "lithologicalDomainOther is not defined properly (lm_type cannot be NA)"
            
            assert np.prod(self.x_ranges == lithologicalDomainOther.x_ranges)==1, f"Expected x_ranges = x_ranges from class, but provided {self.x_ranges} != {lithologicalDomainOther.x_ranges}"
            assert np.prod(self.z_ranges == lithologicalDomainOther.z_ranges)==1, f"Expected z_ranges = z_ranges from class, but provided {self.z_ranges} != {lithologicalDomainOther.z_ranges}"
            assert self.layered_matrix.shape == lithologicalDomainOther.layered_matrix.shape, f"Expected same actual grid size, but provided {self.layered_matrix.shape} != {lithologicalDomainOther.layered_matrix.shape}"
            compare_surface_boundaries(self.surface_boundary, lithologicalDomainOther.surface_boundary)

            if not isinstance(lit_id_none_list, list) or not lit_id_none_list or not all(isinstance(item, str) for item in lit_id_none_list):
                raise ValueError("lit_id_none must be a non-empty list of strings.")

            # Initialize the merged array
            self_matrix = self.layered_matrix
            other_matrix = lithologicalDomainOther.layered_matrix

            # Masks to identify non-'none' values
            mask_self_non_none = ~np.isin(self_matrix, lit_id_none_list)
            mask_other_non_none = ~np.isin(other_matrix, lit_id_none_list)

            # Apply the merging rules using numpy's where function
            merged = np.where(mask_other_non_none, other_matrix, self_matrix)  # (replace it with that of other class even if merged already have values, i.e prioritize other class)

            self.layered_matrix = merged
            # lithologicalDomainMerged.lm_type = "merged_layered_matrix"

            overlap_check = np.sum(mask_self_non_none * mask_other_non_none)
            self.overlap = overlap_check>0 #Overlap with merged layers (useful for utils)
            self.utils_description = f'{self.utils_description} + {lithologicalDomainOther.utils_description}'

    def remeshing_layered_matrix(self, del_x_remeshed, del_z_remeshed, interp_method = 'nearest', replace=True):
        """
        Coarsens/refines the layered matrix based on a coarsened coordinate class, optionally replacing the current matrix.

        Args:
            del_x_remeshed, del_z_remeshed: Spacing of remeshed (new) del_x and del_z.
            interp_method (str): The interpolation method to use ('nearest' or 'nearest_up'). Defaults to 'nearest'.
            replace (bool): Whether to replace the current matrix with the coarsened one. Defaults to False.
        """
        
        if self.del_x == del_x_remeshed and self.del_z == del_z_remeshed:
            self_copy = copy.deepcopy(self)  #Makes sure the change in the object is local to this function only.
        else:
            self_copy = copy.deepcopy(self)  #Makes sure the change in the object is local to this function only.
            
            if interp_method!='nearest' and interp_method!='nearest_up':
                raise ValueError(f"interp_method cannot be other than 'nearest' or 'nearest_up' for replacement. Provided {interp_method}")
                
            if self.dim == 1:
                print("Warning: No effect of remeshing on 1D boundary generation (depth only)")

            else:
                remeshed_2D_domain = check_for_remeshing_coordinate_compatibility(self_copy, del_x_remeshed, del_z_remeshed)
                unique_values, int_mapp = np.unique(self.layered_matrix, return_inverse=True)
                int_mapp = int_mapp.reshape(self.layered_matrix.shape)
                int_values = np.arange(len(unique_values))
                remeshed_int_mapp = f.remeshing_2D_matrix(x_old = self.x_ranges, z_old = self.z_ranges, x_new = remeshed_2D_domain.x_ranges, z_new = remeshed_2D_domain.z_ranges, matrix_2d=int_mapp, interp_method=interp_method)
                
                self_copy.update_domain(remeshed_2D_domain)
                self_copy.layered_matrix = unique_values[remeshed_int_mapp.astype(int)]
                self_copy.lm_type = f'remeshed_{self_copy.lm_type}'

        if replace:  #to verify
            # Update `self`'s attributes in-place to reflect `self_copy`
            self.__dict__.update(self_copy.__dict__)
        else:
            return self_copy

class LithologicalDomain3DReadOnly(LithologicalDomain2DFunctions):
    def __init__(self, lithological_domain_dict, boundary_class=None):
        """
        Get the values of all fields for read_only case from dictionary format (from loaded file.)
        """
        self.read_only=True
        span_x, span_z, del_x, del_z = f.coordinate_vars(lithological_domain_dict['x_ranges'], lithological_domain_dict['z_ranges'])
        name = lithological_domain_dict['name']  
        super().__init__(span_x, span_z, del_x, del_z, name)
        if 'gwt_depth' in lithological_domain_dict.keys():
            gwt_depth = lithological_domain_dict['gwt_depth']
        else:
            gwt_depth = None
            
        self.gwt_depth = gwt_depth
        self.layered_matrix = lithological_domain_dict['layered_matrix']
        self.layered_matrix = np.array([s.decode('utf-8') for s in self.layered_matrix]).reshape((len(self.z_ranges), len(self.x_ranges)))
        
        self.lm_type = lithological_domain_dict['lm_type']
        self.n_layers = lithological_domain_dict['n_layers']
        self.overlap = lithological_domain_dict['overlap']
        self.check = False#lithological_domain_dict['check']# Dont know why, but this is not being saved. rather no addn being saved. so for now using check for utils_desc.
        self.added_prefix = lithological_domain_dict['added_prefix']
        self.utils_description = lithological_domain_dict['check']
        self.boundary_class = boundary_class
        
class LithologicalDomain2D(LithologicalDomain2DFunctions):
    def __init__(self, span_x: float, span_z: float, del_x: float, del_z: float, name: str):
        """
        Initializes the LithogolicalDomain2D instance with given spatial limits, and spacing.
        
        Parameters:
        span_x, span_y, span_z : float
            The upper limit for the x, y, and z-coordinate range.
        del_x, del_y, del_z : float
            The spacing interval for x, y, and z-coordinates.
        name: str
            The name of lithologicaldomain
        """
        super().__init__(span_x, span_z, del_x, del_z, name)
    
    def get_matrix_from_boundary(self, BoundaryCreator_class, gwt_depth=None): 
        """
        Generates a layered matrix from the provided boundary creator class.

        Args:
            BoundaryCreator_class: An object that defines the boundary array and related parameters.
        """
        BoundaryCreator_class = copy.deepcopy(BoundaryCreator_class)  #Makes sure the change in the object is local to this function only.
        assert self.lm_type == 'NA', f"ERROR: The variable is already assigned with {self.lm_type}, should be 'NA'."
        
        self.boundary_class = BoundaryCreator_class
        boundary_matrix = BoundaryCreator_class.boundary_array
        
        assert BoundaryCreator_class.span_z == self.span_z, f"Assertion Error: {BoundaryCreator_class.span_z} != {self.span_z}"
        if boundary_matrix.shape[1] != len(self.x_ranges):
            print(f"WARNING: Boundary matrix (Shape = {boundary_matrix.shape}) does not align with spatial_xs (shape: {len(self.x_ranges)} != {boundary_matrix.shape[1]})")
        
        self.n_layers = BoundaryCreator_class.n_layers
        self.layered_matrix = layer_id_faster(boundary_matrix, self.n_layers, self.z_ranges)
        self.layered_matrix = self.layered_matrix.astype(int)
        self.lm_type = 'from_boundary'
        self.overlap = False #Overlap with merged layers (useful for utils)
        self.gwt_depth = gwt_depth
        self.utils_description = 'Boundary'
        
    def add_surface_boundary_to_curr_boundary(self, SurfaceBoundaryCreator_class, method='pile'):
        #method can be either 'pile' or 'erode'
        assert self.surface_boundary is None, "Surface boundary has already been defined"
        assert method in ['pile', 'erode'], f"Methods can only be either 'pile' or 'erode'. Provided: {method}"

        self_copy = copy.deepcopy(self)  #Makes sure the change in the object is local to this function only.
        self_copy.lm_type = 'NA'

        assert SurfaceBoundaryCreator_class.n_layers == 2, f"The number of layers in top boundary class must be 2 (1 interface). Provided {SurfaceBoundaryCreator_class.n_layers} layers"
        self_copy.get_matrix_from_boundary(SurfaceBoundaryCreator_class)
        self_copy.layered_matrix-=1

        assert self.layered_matrix.shape == self_copy.layered_matrix.shape, f"Expected same actual grid size, but provided {self.layered_matrix.shape} != {self_copy.layered_matrix.shape}"

        if method == 'erode':
            self.layered_matrix=(self.layered_matrix*self_copy.layered_matrix).astype(int)
            # self_copy.plot2d()
        else: #'pile'
            self_copy.plot2d()
            original_array = self.layered_matrix
            padding = np.sum(self_copy.layered_matrix == 0, axis=0)
            max_padding = padding.max()
            
            # Step 2: Create a padded array with enough space for all padding
            padded_shape = (original_array.shape[0]+ max_padding, original_array.shape[1])
            padded_array = np.zeros(padded_shape)
            
            # Step 3: Copy the original array into the padded array at the correct positions
            for row_idx in range(original_array.shape[1]):
                padded_array[padding[row_idx]:padding[row_idx]+original_array.shape[0], row_idx] = original_array[:,row_idx]
                
            # Step 4: Slice the padded array to match the original number of rows
            self.layered_matrix = (padded_array[:original_array.shape[0], :]).astype(int)
        self.lm_type = 'surface_boundary_added'
        self.surface_boundary = get_dict_from_surf_boundary(SurfaceBoundaryCreator_class)
        
        
class LithologicalDomain2D_from_Utils2D(LithologicalDomain2DFunctions):
    def __init__(self, span_x: float, span_z: float, del_x: float, del_z: float, name: str, surfaceBoundaryCreator_class=None):
        """
        Initializes the LithogolicalDomain3D instance with given spatial limits, and spacing.
        
        Parameters:
        span_x, span_z : float
            The upper limit for the x, y, and z-coordinate range.
        del_x, del_z : float
            The spacing interval for x, y, and z-coordinates.
        name: str
            The name of lithologicaldomain
        """
        super().__init__(span_x, span_z, del_x, del_z, name)
        if surfaceBoundaryCreator_class is None:
            self.surface_boundary = None
        else:
            self.surface_boundary = get_dict_from_surf_boundary(surfaceBoundaryCreator_class)
            
    def get_matrix_from_utils2d(self, Utils2d_class, utils_new_ref_points2d, rot_angles_in_degrees, added_prefix, allow_z_axis_rotation_only=True): 
        """
        Generates a layered matrix from a 2D utilities class.
        Allows for rotation of the utils matrix in 3D plane, assuming 2D utils is extruded in y-direction infinitely.
        Note: The area in the cross-section will increase.
        
        Args:
            Utils2d_class: 
                An object that provides 2D utility data.
            angles_in_degrees (list of float): 
                Rotation angles [gamma, beta, theta] in degrees.
            added_prefix: Optional
                Optional prefix (Max: 8) to be added to the merged matrix (default is None). Cannot have "_", or numbers, also cannot be "".
            allow_z_axis_rotation_only: boolean
                If allowing rotation about z-axis only.
        """
        assert f.is_valid_prefix(added_prefix), f"added_prefix cannot have '_' or numbers. Cannot be '', or more than 8 lettered. Provided '{added_prefix}'"
        
        # Note utils_3d is already shifted
        Utils2d_class = copy.deepcopy(Utils2d_class)  #Makes sure the change in the object is local to this function only.
        assert self.lm_type == 'NA', f"ERROR: The variable is already assigned with {self.lm_type}, should be 'NA'."
        assert Utils2d_class.shape is True, "Utils2d_class is not defined properly"
                
        #Here the shift can be negative, and 
        assert f.is_close(self.del_x, Utils2d_class.refining_factor * Utils2d_class.del_x_utils), (
            f"Inconsistent 'del_x' value detected: Expected 'del_x' = refining_factor ({Utils2d_class.refining_factor}) × del_x_utils ({Utils2d_class.del_x_utils}) = {Utils2d_class.refining_factor * Utils2d_class.del_x_utils}, but received {self.del_x}.")

        assert f.is_close(self.del_z, Utils2d_class.refining_factor * Utils2d_class.del_z_utils), (
            f"Inconsistent 'del_z' value detected: Expected 'del_z' = refining_factor ({Utils2d_class.refining_factor}) × del_z_utils ({Utils2d_class.del_z_utils}) = {Utils2d_class.refining_factor * Utils2d_class.del_z_utils}, but received {self.del_z}.")

        utils_new_ref_in_grids = np.array([f.get_nearest_centered_grid_point(utils_new_ref_points2d[0],self.del_z,True), f.get_nearest_centered_grid_point(utils_new_ref_points2d[1],self.del_x,True)])
        if self.surface_boundary is not None:
            surface_boundary_array = self.surface_boundary['boundary_array']
            assert self.surface_boundary['del_x'] == self.del_x, f"Expected del_x = del_x from Surface boundary class, but provided {self.del_x} != {self.surface_boundary['del_x']}"

            ## Adjusting shift accounting for surface accordingly
            z_shift_for_surf = adjust_utils_with_surface_boundary(utils_new_ref_in_grids, surface_boundary_array)
            utils_new_ref_in_grids[0] += f.get_nearest_centered_grid_point(z_shift_for_surf,self.del_z,True)
            
        else:
            surface_boundary_array = None
            
        ## Making sure ref_coord2d lies in one of the grid point so that there is minimum distortion.
        ref_coord2d_in_grid = f.get_nearest_ref_point_in_grid_from_utilsgrid(Utils2d_class, self.del_x, self.del_z)
        shift_points2d_in_grid = utils_new_ref_in_grids - ref_coord2d_in_grid
        shift_points2d = [f.get_centered_grid_point_from_index(shift_points2d_in_grid[0], self.del_z),
                          f.get_centered_grid_point_from_index(shift_points2d_in_grid[1], self.del_x)]
        rot_flag = utils_3d_f.check_rotation_angles(rot_angles_in_degrees)
        
        if rot_flag:
            layered_matrix_coord_table, layered_matrix_grid_table, utils_addn_description = utils_3d_f.get_rotated_utils_grid(Utils2d_class, shift_points2d, rot_angles_in_degrees, allow_z_axis_rotation_only)
        else:
            layered_matrix_coord_table, layered_matrix_grid_table = utils_3d_f.get_table_from_utils2d(Utils2d_class, shift_points2d)
            layered_matrix_coord_table, layered_matrix_grid_table = layered_matrix_coord_table[:, 2:], layered_matrix_grid_table[2:] # first two points in 2D is actually reference points. *y_plot because 2d shapes repeated over y_direction.
            # Assert all elements in the 2nd row are 0
            assert np.all(layered_matrix_coord_table[1, :] == 0), "All elements in row 1 must be 0"
            # Extract rows 0 and 2
            layered_matrix_coord_table = layered_matrix_coord_table[[0, 2], :]
            utils_addn_description = ''
            
        # interp = scipy.interpolate.NearestNDInterpolator(layered_matrix_coord_table.T, layered_matrix_grid_table)
        # print(layered_matrix_coord_table.T, layered_matrix_grid_table)
        Z, X = np.meshgrid(self.z_ranges, self.x_ranges, indexing='ij')
        points_orig = np.vstack([Z.ravel(), X.ravel()]).T  # Shape: (M, 3)
        tree = scipy.spatial.cKDTree(layered_matrix_coord_table.T)
        _, idx = tree.query(points_orig)
        vals_back = layered_matrix_grid_table[idx].reshape(Z.shape)  # shape: same as original
        self.layered_matrix = vals_back.astype(int)  # if not already int   
        # val = interp(Z, X)
        # self.layered_matrix = val.astype(int)
        lithologicalDomain_matrix = self.layered_matrix

        if added_prefix is not None:
            lithologicalDomain_matrix = np.vectorize(lambda x: f"{added_prefix}_{x}")(lithologicalDomain_matrix)
            mask_b_nonzero = (lithologicalDomain_matrix == f"{added_prefix}_0")
            lithologicalDomain_matrix = np.where(mask_b_nonzero, 'X', lithologicalDomain_matrix)  
            self.added_prefix = True
            
        self.layered_matrix = lithologicalDomain_matrix        
        self.lm_type = 'from_utils'
        self.overlap = False #Overlap with merged layers (useful for utils)
        self.utils_description = Utils2d_class.description + utils_addn_description

def get_dict_from_surf_boundary(surfaceBoundaryCreator_class):
    assert surfaceBoundaryCreator_class.boundary_array.shape[0]==1, f"surface boundaries must have only one interfaces. Found {surfaceBoundaryCreator_class.boundary_array.shape[0]}"
    assert surfaceBoundaryCreator_class.n_layers == 2, f"The number of layers in top boundary class must be 2 (1 interface). Provided {surfaceBoundaryCreator_class.n_layers} layers"
    surface_boundary = {'span_x': surfaceBoundaryCreator_class.span_x,
                        'del_x': surfaceBoundaryCreator_class.del_x,
                        'span_z': surfaceBoundaryCreator_class.span_z,
                        'del_z': surfaceBoundaryCreator_class.del_z,
                        'boundary_array': surfaceBoundaryCreator_class.boundary_array}
    
    return surface_boundary

def compare_surface_boundaries(dict1, dict2):
    
    if dict1 is None and dict2 is None:
        return True
    
    # Check if both dicts have the same keys
    if dict1.keys() != dict2.keys():
        return False

    # Compare each value
    for key in dict1:
        val1 = dict1[key]
        val2 = dict2[key]

        if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
            if not np.array_equal(val1, val2):
                return False
        else:
            if val1 != val2:
                return False

    return True
        
def layer_id_faster(processed_boundary, n_layers, spatial_z_ranges):
    """
    Assigns layer IDs to soil points based on processed boundary values and spatial depth ranges.

    Args:
        processed_boundary (np.ndarray): 
            Boundary values for each layer (z,x format).
        n_layers (int): 
            Number of layers.
        spatial_z_ranges (np.ndarray): 
            Depth values defining layer boundaries.

    Returns:
        np.ndarray: Matrix containing assigned layer IDs.
    """
    # Will be layered 1,2,3...
    # Initializing all soil points are last layers
    layer_matrix = np.full((len(spatial_z_ranges), processed_boundary.shape[1]), n_layers)

    compare_matrix = np.ones_like(layer_matrix)*spatial_z_ranges.T[:, None]
    # print(compare_matrix)
    processed_boundary[processed_boundary <= 0] = -1
    processed_boundary[processed_boundary >= spatial_z_ranges[-1]] = spatial_z_ranges[-1]+1
    
    for i in range(n_layers-1):
        boundary_matrix = np.tile(processed_boundary[i,:], (len(spatial_z_ranges), 1))
        # print(boundary_matrix)
        layer_matrix-=(boundary_matrix>=compare_matrix)

    return layer_matrix

def adjust_utils_with_surface_boundary(utils_ref_in_grids, surface_boundary_array):  
    z_shift = 0
    if surface_boundary_array is not None:
        surf_z_max_grid, surf_x_max_grid = surface_boundary_array.shape
        assert surf_z_max_grid == 1, "Surface boundary class can only have one interface."
        # Find the x_coordinate
        if utils_ref_in_grids[1] <= 0:
            ref_x = 0
        elif utils_ref_in_grids[1] >= surf_x_max_grid:
            ref_x = surf_x_max_grid
        else:
            ref_x = utils_ref_in_grids[1]      
        
        # find the y_shift
        z_shift = surface_boundary_array[0,int(ref_x)]
    return z_shift



