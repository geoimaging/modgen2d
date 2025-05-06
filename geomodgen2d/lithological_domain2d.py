# This file is part of geomodgen2D a Python package for ...
# Copyright (C) XXXX Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
# LICENSE

"""Define a three-dimensional domain that defines lithology."""

import numpy as np
import matplotlib.pyplot as plt
import geomodgen2d.general_functions as f
import geomodgen2d.utils_2d as utils_2d
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
        super().__init__(span_x, span_z, del_x, del_z)
        self.lm_type = 'NA'
        self.surface_boundary = None
        self.added_prefix = False
        self.utils_merged = False
        
# Merged lithological

    def print(self):
        print(f"N_z_coord = {self.layered_matrix.shape[0]}, N_x_coord = {self.layered_matrix.shape[1]}")
        print("Layered Matrix : \n", self.layered_matrix) 
        
    def plot2d(self, ax=None, 
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
                if prefix == 'def' and f.is_integer_value(value):
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
        cax = ax.imshow(integer_mapped_array, cmap=fixed_cmap, extent=[x_ranges_plt[0], x_ranges_plt[1], self.span_z, 0], interpolation='nearest')
        ax.set(xlim=x_ranges_plt, ylim = [self.span_z, 0])
        # Create a custom legend
        handles = [plt.Line2D([0], [0], marker='s', color=color_mapping[value], markersize=10, linestyle='') for value in unique_values]
        ax.legend(handles, unique_values, title="Legend", bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.axis('scaled')
        ax.set(
            xlabel='X',
            ylabel='Z',
        )
        return ax

    def remeshing_layered_matrix(self, span_x_remeshed, span_z_remeshed, del_x_remeshed, del_z_remeshed, interp_method = 'nearest', replace=True):
        """
        Coarsens/refines the layered matrix based on a coarsened coordinate class, optionally replacing the current matrix.

        Args:
            span_x_remeshed, span_z_remeshed, del_x_remeshed, del_z_remeshed: A class that contains the coarsened spatial ranges and limits (x, z).
            interp_method (str): The interpolation method to use ('nearest' or 'nearest_up'). Defaults to 'nearest'.
            replace (bool): Whether to replace the current matrix with the coarsened one. Defaults to False.
        """
        self_copy = copy.deepcopy(self)  #Makes sure the change in the object is local to this function only.
        if interp_method!='nearest' and interp_method!='nearest_up':
            raise ValueError(f"interp_method cannot be other than 'nearest' or 'nearest_up' for replacement. Provided {interp_method}")
            
        if self.dim == 1:
            print("Warning: No effect of remeshing on 1D boundary generation (depth only)")

        else:
            remeshed_2D_domain = check_for_remeshing_coordinate_compatibility(self_copy, span_x_remeshed, span_z_remeshed, del_x_remeshed, del_z_remeshed)
            unique_values, int_mapp = np.unique(self.layered_matrix, return_inverse=True)
            int_mapp = int_mapp.reshape(self.layered_matrix.shape)
            int_values = np.arange(len(unique_values))
            
            remeshed_int_mapp = f.remeshing_2D_matrix(x_old = self.x_ranges, z_old = self.z_ranges, x_new = remeshed_2D_domain.x_ranges, z_new = remeshed_2D_domain.z_ranges, matrix_2d=int_mapp, interp_method=interp_method)
            
            self_copy.layered_matrix = unique_values[remeshed_int_mapp.astype(int)]
            self_copy._x_ranges = remeshed_2D_domain.x_ranges
            self_copy._z_ranges = remeshed_2D_domain.z_ranges
            self_copy.lm_type = f'remeshed_{self_copy.lm_type}'

        if replace:  #to verify
            # Update `self`'s attributes in-place to reflect `self_copy`
            self.__dict__.update(self_copy.__dict__)
        else:
            return self_copy
            # unique_values[coarse_int_mapp.astype(int)] to get the replaced_one
            # return unique_values, coarse_int_mapp, coarsened_coordinate_checked_class.x_ranges, coarsened_coordinate_checked_class.z_ranges

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
        
        self.boundary_matrix = BoundaryCreator_class.boundary_array
        
        assert BoundaryCreator_class.span_z == self.span_z, f"Assertion Error: {BoundaryCreator_class.span_z} != {self.span_z}"
        if self.boundary_matrix.shape[1] != len(self.x_ranges):
            print(f"WARNING: Boundary matrix (Shape = {self.boundary_matrix.shape}) does not align with spatial_xs (shape: {len(self.x_ranges)} != {self.boundary_matrix.shape[1]})")
        
        self.n_layers = BoundaryCreator_class.n_layers
        self.layered_matrix = layer_id_faster(self.boundary_matrix, self.n_layers, self.z_ranges)
        self.layered_matrix = self.layered_matrix.astype(int)
        self.lm_type = 'from_boundary'
        self.overlap = False #Overlap with merged layers (useful for utils)
        self.gwt_depth = gwt_depth

    def add_surface_boundary_to_curr_boundary(self, SurfaceBoundaryCreator_class, method='pile'):
        #method can be either 'pile' or 'erode'
        assert self.surface_boundary is None, "Surface boundary has already been defined"
        assert self.utils_merged == False, "Surface boundary must be added before the merging of utils."
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
        

    def merge_with_Utils2D_domains(self, list_lithologicalDomainFromUtils2D = None): #2D NOT CORRECTED YET
        
        ## TO DO: Make sure this surface boundary and surface boundaries of all utils are exactly same.
        
        if list_lithologicalDomainFromUtils2D is not None:
            #assert list of classes
            for i in range(len(list_lithologicalDomainFromUtils2D)):
                utils_lithological = list_lithologicalDomainFromUtils2D[i]
                if self.surface_boundary is not None:
                    assert utils_lithological.span_x == self.surface_boundary['span_x'], "Utils lithological span_x and surface boundary's span_x does not match"
                    assert utils_lithological.span_z == self.surface_boundary['span_z'], "Utils lithological span_z and surface boundary's span_z does not match"
                    assert utils_lithological.del_x == self.surface_boundary['del_x'], "Utils lithological del_x and surface boundary's del_x does not match"
                    assert utils_lithological.del_z == self.surface_boundary['del_z'], "Utils lithological del_z and surface boundary's del_z does not match"
                                    
                if i==0:
                    mergedUtilsDomain = utils_lithological
                else:
                    mergedUtilsDomain = merge_two_lithological_domain(mergedUtilsDomain, utils_lithological, added_prefix=None)

            if len(list_lithologicalDomainFromUtils2D)!=0:
                # Merge the current object (`self`) with the merged utils domain
                merged_self = merge_two_lithological_domain(self, mergedUtilsDomain, added_prefix='U')

        # Update `self`'s attributes in-place to reflect `merged_self`
        self.__dict__.update(merged_self.__dict__)
        self.utils_merged = True
        
        
class LithologicalDomain2D_from_Utils2D(LithologicalDomain2DFunctions):
    def __init__(self, span_x: float, span_z: float, del_x: float, del_z: float, surfaceBoundaryCreator_class=None, name: str=""):
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
        if surfaceBoundaryCreator_class is None:
            self.surface_boundary = None
        else:
            assert surfaceBoundaryCreator_class.boundary_array.shape[0]==1, f"surface boundaries must have only one interfaces. Found {surfaceBoundaryCreator_class.boundary_array.shape[0]}"
            self.surface_boundary = {'span_x': surfaceBoundaryCreator_class.span_x,
                                 'del_x': surfaceBoundaryCreator_class.del_x,
                                 'span_z': surfaceBoundaryCreator_class.span_z,
                                 'del_z': surfaceBoundaryCreator_class.del_z,
                                 'boundary_array': surfaceBoundaryCreator_class.boundary_array}
        
    def get_matrix_from_utils2d(self, Utils2d_class, utils_new_ref): #3d
        """
        Generates a layered matrix from a 2D utilities class.

        Args:
            Utils2d_class: An object that provides 2D utility data.
            utils_new_ref: [z, x] coordinate for utils2d.
        """
        # Note utils_2d is already shifted
        Utils2d_class = copy.deepcopy(Utils2d_class)  #Makes sure the change in the object is local to this function only.
        assert self.lm_type == 'NA', f"ERROR: The variable is already assigned with {self.lm_type}, should be 'NA'."
                
        #Here the shift can be negative, and 
        assert Utils2d_class.shape is True, "Utils2d_class is not defined properly"

        del_z = self.z_ranges[1] - self.z_ranges[0]        
        assert Utils2d_class.del_z == del_z, f"Expected del_z = del_z from class, but provided {Utils2d_class.del_z} != {del_z}"
            
        if Utils2d_class.dim==1:
            del_x = 1
        else:            
            del_x = self.x_ranges[1] - self.x_ranges[0]
            assert Utils2d_class.del_x == del_x, f"Expected del_x = del_x from class, but provided {Utils2d_class.del_x} != {del_x}"
            if self.surface_boundary is not None:
                assert self.surface_boundary['del_x'] == del_x, f"Expected del_x = del_x from Surface boundary class, but provided {self.surface_boundary['del_x']} != {del_x}"

        if self.surface_boundary is not None:
            surface_boundary_array = self.surface_boundary['boundary_array']
        else:
            surface_boundary_array = None
            
        utils_new_ref_in_grid = np.array([utils_new_ref[0]//del_z, utils_new_ref[1]//del_x])
        z_shift_for_surf = adjust_utils_with_surface_boundary(utils_new_ref_in_grid, surface_boundary_array)
        print('z_shift_for_surf:')
        print(z_shift_for_surf, z_shift_for_surf//del_z)
        utils_new_ref_in_grid[0] += z_shift_for_surf//del_z 
        shift = utils_new_ref_in_grid - Utils2d_class.ref_coord
        
        print('shift:')
        print(shift)
        if Utils2d_class.dim!=1:
            Utils2d_class.shift_grid_one_axis(shift_axis='x', shift_in_grid=shift[1], allow_negative_shift = True)
        Utils2d_class.shift_grid_one_axis(shift_axis='z', shift_in_grid=shift[0], allow_negative_shift = True)
        Utils2d_class.expand_grid(len(self.z_ranges), len(self.x_ranges))
        self.layered_matrix = Utils2d_class.grid.astype(int)
        self.lm_type = 'from_utils'
        self.overlap = False #Overlap with merged layers (useful for utils)

def merge_two_lithological_domain(lithologicalDomainA, lithologicalDomainB, added_prefix=None):
    """
    Merges two lithological_domains.

    Parameters:
        lithologicalDomainA, lithologicalDomainB: Two lithologicalDomains to merge.
        
        added_prefix: Optional prefix to be added to the merged matrix (default is None).

    Returns:
        Merged layered matrix object (both original, and other).

    Internally, updates the current layered_matrix after merging.
    """
    lithologicalDomainMerged = copy.deepcopy(lithologicalDomainA)  #Makes sure the change in the object is local to this function only.
    lithologicalDomainB = copy.deepcopy(lithologicalDomainB)  #Makes sure the change in the object is local to this function only.
    assert lithologicalDomainA.lm_type!="NA", "LayeredMatrix_class is not defined properly (lm_type cannot be NA)"
    assert lithologicalDomainB.lm_type!="NA", "lithologicalDomainB is not defined properly (lm_type cannot be NA)"
    assert lithologicalDomainB.added_prefix==False, "lithologicalDomainB cannot have added_prefix (i.e. previously merged with added_prefix), i.e. its layered_matrix must be integers, not strings"
    
    assert np.prod(lithologicalDomainA.x_ranges == lithologicalDomainB.x_ranges)==1, f"Expected del_x = del_x from class, but provided {lithologicalDomainA.del_x} != {lithologicalDomainB.del_x}"
    assert np.prod(lithologicalDomainA.z_ranges == lithologicalDomainB.z_ranges)==1, f"Expected del_z = del_z from class, but provided {lithologicalDomainA.del_z} != {lithologicalDomainB.del_z}"
    assert lithologicalDomainA.layered_matrix.shape == lithologicalDomainB.layered_matrix.shape, f"Expected same actual grid size, but provided {lithologicalDomainA.layered_matrix.shape} != {lithologicalDomainB.layered_matrix.shape}"

    # Initialize the merged array
    lithologicalDomainA_matrix = lithologicalDomainA.layered_matrix
    other_matrix = lithologicalDomainB.layered_matrix
    mask_a_nonzero = (lithologicalDomainA.layered_matrix != 0) #Overlap checks

    if added_prefix is not None:
        other_matrix = np.vectorize(lambda x: f"{added_prefix}_{x}")(other_matrix)
        lithologicalDomainA_matrix = np.vectorize(lambda x: f"{x}")(lithologicalDomainA_matrix)
        lithologicalDomainMerged.added_prefix = True #i.e. converted to string
        mask_b_nonzero = (other_matrix != f"{added_prefix}_0")
    else:
        mask_b_nonzero = (other_matrix != 0)
       
    # Apply the merging rules using numpy's where function
    merged = np.where(mask_b_nonzero, other_matrix, lithologicalDomainA_matrix)  # (replace it with that of other class even if merged already have values, i.e prioritize other class)

    lithologicalDomainMerged.layered_matrix = merged
    # lithologicalDomainMerged.lm_type = "merged_layered_matrix"

    overlap_check = np.sum(mask_a_nonzero * mask_b_nonzero)
    lithologicalDomainMerged.overlap = overlap_check>0 #Overlap with merged layers (useful for utils)
    return lithologicalDomainMerged

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

def adjust_utils_with_surface_boundary(utils_ref_in_grid, surface_boundary_array):  
    z_shift = 0
    if surface_boundary_array is not None:
        surf_z_max_grid, surf_x_max_grid = surface_boundary_array.shape
        assert surf_z_max_grid == 1, "Surface boundary class can only have one interface."
        # Find the x_coordinate
        if utils_ref_in_grid[1] <= 0:
            ref_x = 0
        elif utils_ref_in_grid[1] >= surf_x_max_grid:
            ref_x = surf_x_max_grid
        else:
            ref_x = utils_ref_in_grid[1]      
        
        
        # find the y_shift
        z_shift = surface_boundary_array[0,int(ref_x)]
    return z_shift