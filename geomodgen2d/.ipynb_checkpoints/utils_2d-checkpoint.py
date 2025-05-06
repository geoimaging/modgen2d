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
    def __init__(self, del_x, del_z):
        """
        Initializes the utility shape class.
        
        Parameters:
        del_x (float): Grid resolution in the x-direction. Must be 0 for one-dimensional utils (utils_1d)
        del_z (float): Grid resolution in the z-direction.
        """
        assert del_x>=0, "del_x must be zero(1D) or higher"
        assert del_z>0, "del_z must be larger than 0"
        
        self.shape = False #Initialized No shape
        self.ref_coord = np.array([np.nan,np.nan]) # Reference coordinates [x, y]
        self.del_x = del_x
        self.del_z = del_z
        if self.del_x == 0:
            self.dim = 1
        else:
            self.dim = 2
        self.grid = None
      
    def circular_2d(self, r, util_id = 1, ref='mid'):
        """
        Generates a 2D grid representing a circle of radius r.
        The grid is filled with 1s inside the circle and 0s outside.

        Parameters:
        r (float): Radius of the circle.
        util_id (int, optional): Identifier value for the circle (default is 1).
        ref (str, optional): Reference position ('mid' for center, 'top' for top-aligned).
        
        """
        assert self.shape is False, "ERROR: utils shape has already been defined"
        assert self.dim == 2, "Circular_2d shape is only for 2D model, not 1D as defined"
        assert r>0, "r must be positive"
        util_id = check_util_id(util_id)
        
        # Determine the number of grid points based on the radius and grid resolution (del_x, del_z)
        n_grid_x = int(np.round(2 * r / self.del_x,0))+1
        n_grid_z = int(np.round(2 * r / self.del_z,0))+1

        # Create a blank 2D array (grid) of zeros
        grid = np.zeros((n_grid_z, n_grid_x), dtype=int)

        # Compute the center of the grid (Approx)
        center_x = n_grid_x // 2
        center_z = n_grid_z // 2

        for i in range(n_grid_x):
            for j in range(n_grid_z):
                # Calculate the distance of each point from the center
                dist = np.sqrt((i * self.del_x - r) ** 2 + ((j * self.del_z - r) ** 2))
                if np.round(dist, 4) <= r:
                    grid[j, i] = util_id  # Mark points inside the circle as 1

        # Update object attributes
        self.grid = grid
        if ref == 'mid':
            self.ref_coord = np.array([center_z, center_x])
        elif ref == 'top':
            self.ref_coord = np.array([0, center_x])
        else:
            raise ValueError

        self.shape = True
        
    def rectangle_2d(self, lx, lz, util_id=1, ref='mid'):
        """
        Generates a 2D grid representing a rectangle of size lx x lz.
        The grid is filled with 1s inside the rectangle and 0s outside.

        Parameters:
        lx (float): Length of the rectangle in the x-direction. (lx == 0, for 1D)
        lz (float): Length of the rectangle in the z-direction.
        util_id (int, optional): Identifier value for the rectangle (default is 1).
        ref (str, optional): Reference position ('mid' for center, 'top' for top-aligned).
        """
        assert self.shape is False, "ERROR: utils shape has already been defined"
        if self.dim == 1 and lx!=0:
            raise AssertionError("rectangle_2d shape is only for 2D model, for 1D, lx==0")
        assert lx>=0, "lx must be positive"
        assert lz>0, "lz must be positive"
        util_id = check_util_id(util_id)
        
        # Create a blank 2D array (grid) of zeros
        if lx!=0:
            n_grid_x = int(np.round(lx/self.del_x,0))+1
            mid_grid_x = n_grid_x//2
        else:
            n_grid_x = 1
            mid_grid_x = 0
        
        n_grid_z = int(np.round(lz/self.del_z,0))+1
        if (n_grid_x-1)*self.del_x != lx or (n_grid_z-1) != lz/self.del_z:
            print(f"Adjusted size of rectangle is {(n_grid_x-1)*self.del_x} x {(n_grid_z-1)*self.del_z}, changed from {lx}x{lz}")
            
        grid = np.ones((n_grid_z, n_grid_x), dtype=int)*util_id
        # Mark the region inside the square with 1s
        # grid[:,:] = util_id

        self.grid = grid
        if ref == 'mid':
            self.ref_coord = np.array([(n_grid_z//2), mid_grid_x])
        elif ref == 'top':
            self.ref_coord = np.array([0, mid_grid_x])
        else:
            raise ValueError

        self.shape = True

    def utils_1d(self, lz, util_id=1):
        """
        Generates a 1D grid representing a 1D line of size lz.
        The grid is filled with 1s inside the rectangle and 0s outside.

        Parameters:
        lz (float): Length of the utils in the z-direction.
        util_id (int, optional): Identifier value for the rectangle (default is 1).
        ref is always top
        """
        assert self.del_x == 0, f"Utils_shape.utils_1d is only for 1D model generation, i.e. del_x for utils must be 0 (Provided {self.del_x})"
        self.rectangle_2d(lx=0, lz=lz, util_id=util_id, ref="top")

        
class Utils2D(Utils_shapes):
    def __init__(self, del_x, del_z):
        """
        Initializes the utility class inheriting from Utils_shapes.
        
        Parameters:
        del_x (float): Grid resolution in the x-direction. Must be 0 for one-dimensional utils (utils_1d)
        del_z (float): Grid resolution in the z-direction.
        """
        super().__init__(del_x, del_z)

    def shift_grid_both_axes(self, shift_x_in_grid=0, shift_z_in_grid=0):
        """
        Shift reference points by specified grid units.
        
        Parameters:
        shift_x_in_grid (int): Number of grid units to shift in the x-direction (must be non-negative).
        shift_z_in_grid (int): Number of grid units to shift in the z-direction (must be non-negative).
        
        Raises:
        ValueError: If shift_x_in_grid or shift_z_in_grid are not integers or are negative.
        """
        assert self.shape is True, "utils_class is not defined properly"
        if shift_x_in_grid!=0:
            assert self.dim == 2, "Cannot shift in x-grid in 1D model. Only z-direction"
            shift_x_in_grid = f.check_integer(shift_x_in_grid)
            
        if shift_z_in_grid!=0:
            shift_z_in_grid = f.check_integer(shift_z_in_grid)
            
        if shift_x_in_grid < 0 or shift_z_in_grid < 0:
            raise ValueError(f"Shift values must be non-negative. Provided {shift_x_in_grid} and {shift_z_in_grid}")

        # Get original grid dimensions
        original_zs, original_xs = self.grid.shape
    
        # Calculate new grid dimensions
        new_xs = original_xs + shift_x_in_grid
        new_zs = original_zs + shift_z_in_grid
    
        # Create a new grid filled with zeros
        new_grid = np.zeros((new_zs, new_xs), dtype=int)
    
        # Place the original grid in the new grid at the specified shift
        new_grid[shift_z_in_grid:new_zs, shift_x_in_grid:new_xs] = self.grid
    
        self.grid = new_grid
        self.ref_coord = [self.ref_coord[0] + shift_z_in_grid, self.ref_coord[1] + shift_x_in_grid]

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
            original_zs, original_xs = self.grid.shape
        
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
                new_grid[new_zs-original_zs:new_zs, new_xs-original_xs:new_xs] = self.grid
            else:
                if shift_axis == 'x':
                    new_grid[:, 0:new_xs] = self.grid[:, -shift_in_grid:]
                else:
                    new_grid[0:new_zs, :] = self.grid[-shift_in_grid:, :]
        
            self.grid = new_grid
            self.ref_coord = [self.ref_coord[0] + (new_zs-original_zs), self.ref_coord[1] + (new_xs-original_xs)]

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

        self_zlen, self_xlen = self.grid.shape
        if not (self_xlen<=new_grid_xlen and self_zlen<=new_grid_zlen):
            print(f"WARNING: shape (format: z * x) of old grid {self_zlen} x {self_xlen} is greater than {new_grid_zlen} x {new_grid_xlen}. utils might get removed from the model")
        
        new_grid = np.zeros((max(self_zlen, new_grid_zlen), max(self_xlen, new_grid_xlen)), dtype = int)
        new_grid[:self_zlen, :self_xlen] = self.grid
        new_grid = new_grid[:new_grid_zlen, :new_grid_xlen]
        self.grid = new_grid
        self.ref_coord = self.ref_coord

    def scale_shapes(self, scale_factor):
        """
        Scales the shape by a given factor.
        
        Parameters:
        scale_factor (float): Factor to scale the shape. Must be positive.
        """
        if scale_factor <= 0:
            raise ValueError("Scale factor must be a positive number.")

        new_x_len = int(np.round(self.grid.shape[1] * scale_factor,0))
        new_z_len = int(np.round(self.grid.shape[0] * scale_factor,0))

        scaled_grid = np.zeros((new_z_len, new_x_len), dtype=int)
        
        # Populate the scaled grid by mapping original grid values
        for i in range(new_x_len):
            for j in range(new_z_len):
                # Map the scaled grid index to the nearest original grid index
                orig_i = int(i / scale_factor)
                orig_j = int(j / scale_factor)
                scaled_grid[j, i] = self.grid[orig_j, orig_i]

        # Update the grid and resolution
        self.grid = scaled_grid
        self.ref_coord = [i*scale_factor for i in self.ref_coord]
        
    def merge_shapes(self, utils_class_other, ref='mid'):
        """
        Merges two utils_shape into one and updates the reference coordinates.
    
        Args:
            utils_class_other: Another utils object to merge with.
            ref (str): The reference type for the merged grid, either 'mid' or 'top'. Defaults to 'mid'.
    
        Raises:
            ValueError: If the reference type is invalid or if grid spacing is not compatible.
        """
            
        utils_class_other = copy.deepcopy(utils_class_other)
        assert self.shape is True, "utils_class is not defined properly"
        assert utils_class_other.shape is True, "utils_class_other is not defined properly"
        assert self.del_x == utils_class_other.del_x and self.del_z == utils_class_other.del_z, "Code Error: Merged utils must have same spacing"
        
        shift = utils_class_other.ref_coord - self.ref_coord
        if shift[1]>=0 and shift[0]>=0:
            self.shift_grid_both_axes(shift_x_in_grid=shift[1], shift_z_in_grid=shift[0])
        elif shift[1]<=0 and shift[0]<=0:
            utils_class_other.shift_grid_both_axes(shift_x_in_grid=-shift[1], shift_z_in_grid=-shift[0])
        elif shift[1]>=0 and shift[0]<=0:
            self.shift_grid_both_axes(shift_x_in_grid=shift[1], shift_z_in_grid=0)
            utils_class_other.shift_grid_both_axes(shift_x_in_grid=0, shift_z_in_grid=-shift[0])
        elif shift[1]>=0 and shift[0]<=0:
            utils_class_other.shift_grid_both_axes(shift_x_in_grid=-shift[1], shift_z_in_grid=0)
            self.shift_grid_both_axes(shift_x_in_grid=0, shift_z_in_grid=shift[0])
        else:
            raise ValueError('Code Error: Impossible case?')

        # print(self.ref_coord == utils_class_other.ref_coord)
        assert np.prod(self.ref_coord == utils_class_other.ref_coord)==1, f"Code Error: After shifting, coordinates must be same. Provided {self.ref_coord} == {utils_class_other.ref_coord}"

        self_zlen, self_xlen = self.grid.shape
        util2_zlen, util2_xlen = utils_class_other.grid.shape
        # print(shift)
        # print(self_xlen, util2_xlen, self_zlen, util2_zlen)
        new_grid_xlen, new_grid_zlen = np.max([self_xlen, util2_xlen]), np.max([self_zlen, util2_zlen])

        self.expand_grid(new_grid_zlen, new_grid_xlen)
        utils_class_other.expand_grid(new_grid_zlen, new_grid_xlen)

        # Initialize the merged array
        merged = np.zeros_like(self.grid)
    
        mask_a_nonzero = (self.grid != 0)
        mask_b_nonzero = (utils_class_other.grid != 0)
    
        # Apply the merging rules using numpy's where function
        merged = np.where(mask_b_nonzero, utils_class_other.grid, merged)  # b != 0
        merged = np.where(mask_a_nonzero, self.grid, merged)  # a != 0 (replace it with a even if merged already have values, i.e priortize b)
    
        self.grid = merged
        if ref == 'mid':
            self.ref_coord = np.array([(new_grid_zlen//2), (new_grid_xlen//2)])
        elif ref == 'top':
            self.ref_coord = np.array([(new_grid_zlen), (new_grid_xlen//2)])
        else:
            raise ValueError    

    def clear_utils_properties(self, del_x=0, del_z=0):
        """
        clear the inputs for Utils_2D object for redefining.
        """
        if del_x==0 and del_z==0:
            self.__init__(self.del_x, self.del_z)
        else:
            self.__init__(self.del_x, del_z)
            
    def plot(self, ax=None,     
         color_map_items = plt.get_cmap('Set3', 10)):
        """
        Plots the grid and reference coordinates.
    
        Args:
            ax (matplotlib.axes.Axes, optional): The axis to plot on. If None, a new axis will be created.
            color_map_items (matplotlib.colors.ListedColormap, optional): A colormap to use for plotting. Defaults to 'Set3' if None.
        """

        if self.grid is None:
            raise ValueError("Grid has not been defined.")

        unique_values = np.unique(self.grid)
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
        integer_mapped_array = np.vectorize(int_map.get)(self.grid)
    
        # Create a colormap from the color mapping
        colors = [color_mapping[value] for value in unique_values]
        fixed_cmap = mcolors.ListedColormap(colors)

        if ax is None:
            fig, ax = plt.subplots()
        
        z_lim_grid, x_lim_grid = self.grid.shape
        plot_del_x = self.del_x
        
        if self.dim == 1:
            plot_del_x = self.ref_coord[0]*self.del_z/4
        # xv, yv = np.meshgrid(range(len(self.grid.shape[0])), range(len(self.grid.shape[1])))
        # Plot the grid with a colormap based on the grid values
        c = ax.imshow(self.grid, cmap=fixed_cmap, interpolation='nearest', extent=[0,x_lim_grid*plot_del_x, z_lim_grid*self.del_z,0])
        ax.scatter(self.ref_coord[1]*plot_del_x, self.ref_coord[0]*self.del_z, color='red', label='Reference Coordinate', zorder=5)
        
        # Create a custom legend
        handles = [plt.Line2D([0], [0], marker='s', color=color_mapping[value], markersize=10, linestyle='') for value in unique_values]
        ax.legend(handles, unique_values, title="Legend", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        ax.set_title('Grid Visualization')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Z Coordinate')
        
        return ax

def check_util_id(util_id):
    if not f.is_integer_value(util_id) or util_id<=0:
        raise ValueError(f"Invalid util_id, must be positive integer. Provided {util_id}")
    else:
        util_id = f.check_integer(util_id)
    return util_id

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