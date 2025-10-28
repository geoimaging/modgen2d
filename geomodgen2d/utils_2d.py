# This file is part of geomodgen3d a Python package for ...
# Copyright (C) XXXX Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
# LICENSE

## Note: Added by Sanish (Feb 24, 2025)

import numpy as np
import geomodgen2d.general_functions as f
import copy
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

class Utils_shapes:
    def __init__(self, del_x, del_z, refining_factor:int=1):
        """
        Initializes the utility shape class.
        
        Parameters:
        del_x (float): 
            Grid resolution in the x-direction. Must be 0 for one-dimensional utils (utils_1d)
        del_z (float): 
            Grid resolution in the z-direction.
        refining_factor (int): 
            Refining factor of resolution for better nearest interpolation later in lit_domain.
        """
        assert del_x>=0, "del_x must be zero(1D) or higher"
        assert del_z>0, "del_z must be larger than 0"
        assert refining_factor>=1, "refining_factor must be an positive integer."
        assert f.check_integer(refining_factor), "refining_factor must be an positive integer."
        
        self.shape = False #Initialized No shape
        self.ref_coord2d_in_utilsgrid = np.array([np.nan,np.nan]) # Reference coordinates [x, y]
        self.del_x_utils = del_x/refining_factor
        self.del_z_utils = del_z/refining_factor
        self.refining_factor = refining_factor
        if self.del_x_utils == 0:
            self.dim = 1
        else:
            self.dim = 2
        self.utilsgrid = None
        self.description = ''
      
    def circular_2d(self, d, util_id = 1, warn_adjustments=False):
        """
        Generates a 2D grid representing a circle of radius r.
        The grid is filled with 1s inside the circle and 0s outside.
        Reference position is always at x=r, z=0.

        Parameters:
        d (float): 
            Diameter of the circle.
        util_id (int, optional): 
            Identifier value for the circle (default is 1).
        warn_adjustments (bool, optional):
            Print warnings for adjustments of radius and reference points done for better discretization. 
        """
        assert self.shape is False, "ERROR: utils shape has already been defined"
        assert self.dim == 2, "Circular_2d shape is only for 2D model, not 1D as defined"
        assert d>0, "d must be positive"
        util_id = f.check_util_id(util_id)
        
        del_x = self.del_x_utils * self.refining_factor
        
        ## Adjustment for better discretization
        old_act_d = d
        d = old_act_d - del_x/2 * 2#Actual r is larger because of discrete points are center of pixel, and the distance from its center to edge is del_x/2 in x_direction.
        n_grid_d = int(np.round(d/del_x,0))+1
        d = (n_grid_d-1)*del_x 
        act_d = d + del_x/2 * 2
        
        if warn_adjustments and not f.is_close(old_act_d, act_d):
            print(f"Adjusted radius of circle is {act_d}, changed from {old_act_d}")
        
        # Determine the number of grid points based on the radius and grid resolution (del_x, del_z)
        n_utilsgrid_x = int(np.round(d / self.del_x_utils,0))+1
        n_utilsgrid_z = int(np.round(d / self.del_z_utils,0))+1
        
        # Create a blank 2D array (grid) of zeros
        grid = np.zeros((n_utilsgrid_z, n_utilsgrid_x), dtype=int)

        r = d/2
        for i in range(n_utilsgrid_x):
            for j in range(n_utilsgrid_z):
                # Calculate the distance of each point from the center
                dist = np.sqrt((i * self.del_x_utils - r) ** 2 + ((j * self.del_z_utils - r) ** 2))
                if np.round(dist, 4) <= r:
                    grid[j, i] = util_id  # Mark points inside the circle as 1

        # Update object attributes
        self.utilsgrid = grid
        
        # Compute the center of the grid (Approx)
        center_x = (n_utilsgrid_x-1) // 2 
        self.ref_coord2d_in_utilsgrid = np.array([0, center_x])
        
        self.description = f'Circular 2d of diameter {act_d}'
        self.shape = True
        
    def rectangle_2d(self, lx, lz, util_id=1, warn_adjustments=False):
        """
        Generates a 2D grid representing a rectangle of size lx x lz.
        The grid is filled with 1s inside the rectangle and 0s outside.

        Parameters:
        lx (float): 
            Length of the rectangle in the x-direction. (lx == 0, for 1D)
        lz (float): 
            Length of the rectangle in the z-direction.
        util_id (int, optional): 
            Identifier value for the rectangle (default is 1).
        warn_adjustments (bool, optional):
            Print warnings for adjustments of radius and reference points done for better discretization. 
        """
        assert self.shape is False, "ERROR: utils shape has already been defined"
        if self.dim == 1 and lx!=0:
            raise AssertionError("rectangle_2d shape is only for 2D model, for 1D, lx==0")
        assert lx>=0, "lx must be positive"
        assert lz>0, "lz must be positive"
        util_id = f.check_util_id(util_id)
        
        del_x = self.del_x_utils * self.refining_factor
        del_z = self.del_z_utils * self.refining_factor
        
        ## Adjustment for better discretization
        old_act_lx = lx
        old_act_lz = lz
        
        lz = old_act_lz - del_z/2 * 2 #Actual lz is larger because of discrete points are center of pixel, and the distance from its center to edge is del_z/2 in z_direction.
        n_grid_z = int(np.round(lz/del_z,0))+1
        lz = (n_grid_z-1)*del_z 
        act_lz = lz + del_z/2 * 2
        
        if lx!=0:
            lx = old_act_lx - del_x/2 * 2
            n_grid_x = int(np.round(lx/del_x,0))+1
            lx = (n_grid_x-1)*del_x
            act_lx = lx + del_x/2 * 2  
            n_utilsgrid_x = int(np.round(lx / self.del_x_utils,0))+1
            mid_grid_x = (n_utilsgrid_x-1)//2
            
        else:
            n_grid_x = 1
            n_utilsgrid_x = 1
            mid_grid_x = 0 # Or 1?
        
        if warn_adjustments:
            if not f.is_close(act_lx, old_act_lx) or not f.is_close(act_lz, old_act_lz):
                print(f"Adjusted size of rectangle is {act_lz} x {act_lx}, changed from {old_act_lz} x {old_act_lx}")
            
        # Determine the number of grid points based on the radius and grid resolution (del_x, del_z)
        n_utilsgrid_z = int(np.round(lz / self.del_z_utils,0))+1
        
        grid = np.ones((n_utilsgrid_z, n_utilsgrid_x), dtype=int)*util_id  

        self.utilsgrid = grid
        self.ref_coord2d_in_utilsgrid = np.array([0, mid_grid_x])

        self.description = f'Rectangular 2d of size (lz x lx) = ({act_lz:.6g} x {act_lx:.6g})'
        self.shape = True

    def utils_1d(self, lz, util_id=1):
        """
        Generates a 1D grid representing a 1D line of size lz.
        The grid is filled with 1s inside the rectangle and 0s outside.

        Parameters:
        lz (float): 
            Length of the utils in the z-direction.
        util_id (int, optional): 
            Identifier value for the rectangle (default is 1).
        ref is always top
        """
        assert self.del_x_utils == 0, f"Utils_shape.utils_1d is only for 1D model generation, i.e. del_x for utils must be 0 (Provided {self.del_x_utils})"
        self.rectangle_2d(lx=0, lz=lz, util_id=util_id, ref="top")

        
class Utils2D(Utils_shapes):
    def __init__(self, del_x, del_z, refining_factor:int=1):
        """
        Initializes the utility class inheriting from Utils_shapes.
        
        Parameters:
        del_x (float): 
            Grid resolution in the x-direction. Must be 0 for one-dimensional utils (utils_1d)
        del_z (float): 
            Grid resolution in the z-direction.        
        refining_factor (int): 
            Refining factor of resolution for better nearest interpolation later in lit_domain.
        """
        super().__init__(del_x, del_z, refining_factor)

    def shift_grid_one_axis(self, shift_axis='x', shift_in_grid=0, allow_negative_shift = False):
        """
        Shift reference points by specified grid units.
        
        Parameters:
        shift_axis: Either x_axis or z_axis
        shift_in_grid (int): Number of grid units to shift.
        allow_negative_shift (bool): If negative shift is allowed or not. (Negative shift means truncation)
        
        Raises:
        ValueError: If shift_x_in_grid or shift_z_in_grid are not integers or are negative (if allow_negative_shift is False).
        """
        assert self.shape is True, "utils_class is not defined properly"
        assert shift_axis in ['x', 'z'], f"shift axis can only be either 'x' or 'z'. Not {shift_axis}"

        if shift_in_grid!=0:
            shift_in_grid = f.check_integer(shift_in_grid)
            
            if not allow_negative_shift: 
                if shift_in_grid < 0:
                    raise ValueError(f"Shift values must be non-negative. Provided {shift_in_grid}")
    
            # Get original grid dimensions
            original_zs, original_xs = self.utilsgrid.shape
        
            # Calculate new grid dimensions
            if shift_axis == 'x':
                assert self.dim == 2, "Cannot shift in x-grid in 1D model. Only z-direction"
                new_xs = original_xs + shift_in_grid
                new_zs = original_zs
            else:
                new_xs = original_xs
                new_zs = original_zs + shift_in_grid
        
            # Create a new grid filled with zeros
            new_grid = np.zeros((new_zs, new_xs), dtype=int)
        
            # Place the original grid in the new grid at the specified shift
            if shift_in_grid>0:
                new_grid[new_zs-original_zs:new_zs, new_xs-original_xs:new_xs] = self.utilsgrid
            else:
                if shift_axis == 'x':
                    new_grid[:, 0:new_xs] = self.utilsgrid[:, -shift_in_grid:]
                else:
                    new_grid[0:new_zs, :] = self.utilsgrid[-shift_in_grid:, :]
        
            self.utilsgrid = new_grid
            self.ref_coord2d_in_utilsgrid = [self.ref_coord2d_in_utilsgrid[0] + (new_zs-original_zs), self.ref_coord2d_in_utilsgrid[1] + (new_xs-original_xs)]

    def expand_grid(self, new_grid_zlen, new_grid_xlen):
        """
        Expands (zero_pad or truncate) the grid size to new dimensions.
        
        Parameters:
        new_grid_xlen (int): New length in the x-direction.
        new_grid_zlen (int): New length in the z-direction.
        
        Raises:
        ValueError: If new dimensions are smaller than the current grid.
        """
        assert self.shape is True

        new_grid_xlen = f.check_integer(new_grid_xlen)
        new_grid_zlen = f.check_integer(new_grid_zlen)

        self_zlen, self_xlen = self.utilsgrid.shape
        if not (self_xlen<=new_grid_xlen and self_zlen<=new_grid_zlen):
            print(f"WARNING: shape (format: z * x) of old grid {self_zlen} x {self_xlen} is greater than {new_grid_zlen} x {new_grid_xlen}. utils might get removed from the model")
                    
        new_grid = np.zeros((max(self_zlen, new_grid_zlen), max(self_xlen, new_grid_xlen)), dtype = int)
        new_grid[:self_zlen, :self_xlen] = self.utilsgrid
        new_grid = new_grid[:new_grid_zlen, :new_grid_xlen]
        self.utilsgrid = new_grid
        self.ref_coord2d_in_utilsgrid = self.ref_coord2d_in_utilsgrid
        
    def scale_shapes(self, scale_factor):
        """
        Scales the shape by a given factor.
        
        Parameters:
        scale_factor (float): Factor to scale the shape. Must be positive.
        """
        if scale_factor <= 0:
            raise ValueError("Scale factor must be a positive number.")

        if self.dim != 1:
            new_x_len = int(np.round(self.utilsgrid.shape[1] * scale_factor,0))
        else:
            new_x_len = self.utilsgrid.shape[1]
        new_z_len = int(np.round(self.utilsgrid.shape[0] * scale_factor,0))

        scaled_grid = np.zeros((new_z_len, new_x_len), dtype=int)
        
        # Populate the scaled grid by mapping original grid values
        for i in range(new_x_len):
            for j in range(new_z_len):
                # Map the scaled grid index to the nearest original grid index
                orig_i = int(i / scale_factor)
                orig_j = int(j / scale_factor)
                scaled_grid[j, i] = self.utilsgrid[orig_j, orig_i]

        # Update the grid and resolution
        self.utilsgrid = scaled_grid
        self.ref_coord2d_in_utilsgrid = [i*scale_factor for i in self.ref_coord2d_in_utilsgrid] # To check if it is okay for non-integer scaling
        
        self.description += f', then scaled by factor of {scale_factor}'
        
    def merge_shapes(self, utils_class_other):
        """
        Merges two utils_shape into one and updates the reference coordinates.
    
        Args:
            utils_class_other: Another utils object to merge with.
    
        Raises:
            ValueError: If the reference type is invalid or if grid spacing is not compatible.
        """
            
        utils_class_other = copy.deepcopy(utils_class_other)
        assert self.shape is True, "utils_class is not defined properly"
        assert utils_class_other.shape is True, "utils_class_other is not defined properly"
        assert self.del_x_utils == utils_class_other.del_x_utils and self.del_z_utils == utils_class_other.del_z_utils, "Code Error: Merged utils must have same spacing"
        
        self_zlen, self_xlen = self.utilsgrid.shape
        util2_zlen, util2_xlen = utils_class_other.utilsgrid.shape
        # print(shift)
        # print(self_xlen, util2_xlen, self_zlen, util2_zlen)
        new_grid_xlen, new_grid_zlen = np.max([self_xlen, util2_xlen]), np.max([self_zlen, util2_zlen])

        self.expand_grid(new_grid_zlen, new_grid_xlen)
        utils_class_other.expand_grid(new_grid_zlen, new_grid_xlen)

        # Initialize the merged array
        merged = np.zeros_like(self.utilsgrid)
    
        mask_a_nonzero = (self.utilsgrid != 0)
        mask_b_nonzero = (utils_class_other.utilsgrid != 0)
    
        # Apply the merging rules using numpy's where function
        merged = np.where(mask_b_nonzero, utils_class_other.utilsgrid, merged)  # b != 0
        merged = np.where(mask_a_nonzero, self.utilsgrid, merged)  # a != 0 (replace it with a even if merged already have values, i.e priortize b)
    
        self.utilsgrid = merged
        self.ref_coord2d_in_utilsgrid = np.array([0, (new_grid_xlen//2)])
        
        self.description+= f', then merged with Utils2d of {utils_class_other.description}' 
            
    def clear_utils_properties(self, del_x=None, del_z=None):
        """
        clear the inputs for Utils_2D object for redefining.
        """
        if del_x is None or del_z is None:
            self.__init__(self.del_x_utils, self.del_z_utils)
        else:
            self.__init__(del_x, del_z)
            
    def plot(self, ax=None, discrete_point_size=0, legend = True, title = 'Grid Visualization',
         color_map_items = plt.get_cmap('Set3', 10)):
        """
        Plots the grid and reference coordinates.
    
        Args:
            ax (matplotlib.axes.Axes, optional): The axis to plot on. If None, a new axis will be created.
            color_map_items (matplotlib.colors.ListedColormap, optional): A colormap to use for plotting. Defaults to 'Set3' if None.
        """

        if self.utilsgrid is None:
            raise ValueError("Grid has not been defined.")

        unique_values = np.unique(self.utilsgrid)
        color_mapping = {}
        for value in unique_values:
            try:
                cmap = color_map_items
                index = int(float(value)) % 10
                color_mapping[value] = cmap(index)
                assigned = True

            except:
                color_mapping[value] = "#" + ''.join([np.random.choice(list('0123456789ABCDEF')) for _ in range(6)])

        int_map = {value: idx for idx, value in enumerate(unique_values)}
        integer_mapped_array = np.vectorize(int_map.get)(self.utilsgrid)
    
        # Create a colormap from the color mapping
        colors = [color_mapping[value] for value in unique_values]
        fixed_cmap = mcolors.ListedColormap(colors)

        if ax is None:
            fig, ax = plt.subplots()
        
        z_lim_grid, x_lim_grid = self.utilsgrid.shape
        plot_del_x = self.del_x_utils
        plot_del_z = self.del_z_utils
        
        if self.dim == 1:
            plot_del_x = self.ref_coord[0]*self.del_z_utils/4

        # Plot the grid with a colormap based on the grid values
        c = ax.imshow(self.utilsgrid, cmap=fixed_cmap, interpolation='nearest', 
                      extent=[0-plot_del_x/2,(x_lim_grid-1)*plot_del_x+plot_del_x/2, (z_lim_grid-1)*plot_del_z+plot_del_z/2,0-plot_del_z/2]
                      )
        
        if discrete_point_size!=0:
            x, y = np.meshgrid(np.arange(x_lim_grid), np.arange(z_lim_grid))
            ax.scatter(x.flatten()*plot_del_x, y.flatten()*plot_del_z, c=self.utilsgrid.flatten(), cmap=fixed_cmap, s=discrete_point_size, edgecolors='k') 
        
        ax.scatter(self.ref_coord2d_in_utilsgrid[1]*plot_del_x, self.ref_coord2d_in_utilsgrid[0]*self.del_z_utils, color='red', label='Reference Coordinate', zorder=5)
        # Create a custom legend
        if legend:
            handles = [plt.Line2D([0], [0], marker='s', color=color_mapping[value], markersize=10, linestyle='') for value in unique_values]
            ax.legend(handles, unique_values, title="Legend", bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_title(title)
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Z Coordinate')
        
        return ax
    

class Utils1D(Utils2D):
    def __init__(self, del_z, lz, util_id=1):
        """
        Initializes the 1D utility class inheriting from Utils_shapes.
        
        Parameters:
        lz (float): length of utils (1D)
        del_z (float): Grid resolution in the z-direction.
        """
        super().__init__(0, del_z)
        self.utils_1d(lz, util_id)