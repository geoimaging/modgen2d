# This file is part of modgen2d a Python package for ...
# Copyright (C) XXXX Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
# LICENSE

## Note: Added by Sanish (Feb 24, 2025)

"""2D obstruction geometry utilities. """
import numpy as np
import modgen2d.general_functions as f
import copy
import warnings
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

class _Obstruction2DFunctions:
    """
    Base utilities for 2D obstruction grids.
    
    Parameters
    ----------
    dl : float
        Grid spacing for the obstruction.
    ref_xz_symbolic : list of length 2, optional
        Acceptable values: ['o', 'c'] or ['O', 'C'] or ['0', 'c'], etc. [0, '0', 'O', 'o'] and [1, 'c', 'C']
        Example: ['o','c'] means x is at 0 and z is at center. Note: 'C' means center of grid, which might not be center of obstacles (depends on snap.)
    snap_to_dl : bool, optional
        If True, geometry dimensions are snapped to multiples
        of the grid spacing.
    """
    def __init__(self, dl:float, ref_xz_symbolic = ['c', 'c'], snap_to_dl:bool=True):
        """
        Functions only.
        """
        if dl<=0:
            raise ValueError("Obstacles grid step size must be greater than zero")
        
        if not isinstance(snap_to_dl, bool):
            raise ValueError("snap_to_dl must be a boolean")

        ref_xz_symbolic = self.validate_ref_xz_symbolic_format(ref_xz_symbolic) 

        self.dl = dl
        self.snap_to_dl = snap_to_dl
        self.ref_xz_symbolic = ref_xz_symbolic
        self.center_in_unit_length = None
        self.shape = False #Initialized No shape
        self.grid2d = None
        self.ref_xz_override = None # Reference coordinates [x, z] if manual set 
        self.description = ''
     
    def plot(self, ax=None, discrete_point_size=0, white_edges_size = 0, ref_point_size = 1, legend = True, 
             show_padding = False,
             title = 'Grid Visualization',
             id2label_dict = None, ref_coord_label='Ref. Coord.', legend_title='Legend',
             color_map_items = plt.get_cmap('Set3', 10)):
        """
        Plot the obstruction grid and reference point.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axis to plot on. If None, a new figure is created.
        discrete_point_size : int, optional
            Marker size for plotting discrete grid points.
        legend : bool, optional
            If True, show a legend for obstruction IDs.
        title : str, optional
            Plot title.
        show_padding : bool, optional
            If True, show one layer of zero padding around the grid.
        color_map_items : matplotlib.colors.Colormap, optional
            Colormap used for obstruction IDs.

        Returns
        -------
        matplotlib.axes.Axes
            Axis containing the plot.
        """
        if self.grid2d is None:
            raise ValueError("Grid has not been defined.")
        grid2d = self.grid2d
        extra_grid = 0
        plot_del_x = self.dl
        plot_del_z = self.dl
        
        x_lim_grid, z_lim_grid = grid2d.shape
        extent=[0,(x_lim_grid)*plot_del_x, (z_lim_grid)*plot_del_z,0]
        
        if show_padding:
            grid2d = np.pad(grid2d, pad_width=1, mode='constant', constant_values=0) #Note padded one so ref_coord also add 1 
            extra_grid = 1
            x_lim_grid, z_lim_grid = grid2d.shape
            extent=[-plot_del_x,(x_lim_grid-1)*plot_del_x, (z_lim_grid-1)*plot_del_z,-plot_del_z]
            
        unique_values = np.unique(grid2d)
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
        integer_mapped_array = np.vectorize(int_map.get)(grid2d)
    
        # Create a colormap from the color mapping
        colors = [color_mapping[value] for value in unique_values]
        fixed_cmap = mcolors.ListedColormap(colors)

        if ax is None:
            fig, ax = plt.subplots()
        
        # Plot the grid with a colormap based on the grid values
        c = ax.imshow(grid2d.T, cmap=fixed_cmap, 
                      interpolation='nearest', 
                      extent=extent,
                      )
        
        if discrete_point_size!=0:
            x, y = np.meshgrid(np.arange(x_lim_grid) + 1/2 - extra_grid, np.arange(z_lim_grid) + 1/2 - extra_grid)
            ax.scatter(x.flatten()*plot_del_x, y.flatten()*plot_del_z, c=grid2d.T.flatten(), 
                       cmap=fixed_cmap, 
                       edgecolors='white',  # thin white borders
                       linewidths=0.3,   
                       marker='s',          # square marker
                       s=discrete_point_size, 
                       ) 

        if white_edges_size != 0:
            edges_x = (np.arange(x_lim_grid) - extra_grid) * plot_del_x
            edges_z = (np.arange(z_lim_grid) - extra_grid) * plot_del_z
        
            for e in edges_x:
                ax.axvline(e, color='white', linewidth=white_edges_size)
        
            for e in edges_z:
                ax.axhline(e, color='white', linewidth=white_edges_size)

        ref_xz_in_unit_length = self.get_ref_xz_in_unit_length()
            
        ax.scatter(ref_xz_in_unit_length[0], ref_xz_in_unit_length[1], color='red', s=ref_point_size, label=ref_coord_label, zorder=5)
        # Create a custom legend
        if legend:
            labels = unique_values
            if id2label_dict is not None:
                labels = [id2label_dict[label] if label in id2label_dict else label for label in labels]
                labels = [lbl.decode('utf-8') if isinstance(lbl, bytes) else lbl for lbl in labels]

            handles = [plt.Line2D([0], [0], marker='s', color=color_mapping[value], markersize=10, linestyle='') for value in unique_values]

            # add reference marker
            handles.append(
                plt.Line2D([0], [0], marker='o', color='red',
                           markersize=ref_point_size, linestyle='')
            )
            
            labels = list(labels) + [ref_coord_label]

            ax.legend(handles, labels, title=legend_title, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_title(title)
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Z Coordinate')
        
        return ax
    
    def set_manual_ref_xz(self, ref_xz_override, symbolic=False):
        """
        Set reference coordinates manually.

        Parameters
        ----------
        ref_xz_override : list or tuple of length 2
            Reference coordinates or symbolic definition.
        symbolic : bool, optional
            If True, `ref_xz_override` is interpreted symbolically.
        """
        if symbolic:
            ref_xz_override = self.validate_ref_xz_symbolic_format(ref_xz_override)
            self.ref_xz_override = None
            self.ref_xz_symbolic = ref_xz_override        
        else:
            self.validate_ref_xz_override(ref_xz_override)
            self.ref_xz_override = ref_xz_override
            self.ref_xz_symbolic = None        
        
    @staticmethod
    def validate_ref_xz_symbolic_format(ref_xz_symbolic):
        """
        Validate symbolic reference specification.

        Parameters
        ----------
        ref_xz_symbolic : list or tuple of length 2
            Symbolic reference definition.

        Returns
        -------
        list
            Processed reference specification (0 = origin, 1 = center).

        Raises
        ------
        ValueError
            If the format or symbols are invalid.
        """
        # --- Validate ref_xz_symbolic format ---
        if not isinstance(ref_xz_symbolic, (list, tuple)) or len(ref_xz_symbolic) != 2:
            raise ValueError("ref_xz_symbolic must be a list or tuple of two elements (e.g., ['o', 'c']).")

        valid_symbols = {'o', 'O', '0', 0, 1, 'c', 'C'}
        if not all(elem in valid_symbols for elem in ref_xz_symbolic):
            raise ValueError(
                f"Invalid ref_xz_symbolic values {ref_xz_symbolic}. Allowed values are any combination of 'o', 'O', '0', 0, or 'c'."
            ) 
        
        ref_xz_symbolic_processed = []
        for ref_xz_each in ref_xz_symbolic:
            if ref_xz_each in ['o','O','0',0]:
                ref_xz_symbolic_processed.append(0)
            else:
                ref_xz_symbolic_processed.append(1)
                
        return ref_xz_symbolic_processed
                
    @staticmethod
    def validate_ref_xz_override(ref_xz_override):
        """
        Get reference coordinates in physical units.

        Returns
        -------
        numpy.ndarray, shape (2,)
            Reference coordinates in unit length.
        """
        if not isinstance(ref_xz_override, (list, tuple)) or len(ref_xz_override) != 2:
                raise ValueError("center_coord_in_unit_length must be a list or tuple of two numeric elements (e.g., [z_center, x_center]).")

        if not all(isinstance(val, (int, float)) for val in ref_xz_override):
            raise ValueError("All elements of center_coord_in_unit_length must be numeric (int or float).")
        
    def get_ref_xz_in_unit_length(self):
        ref_val_flag = False
        if self.ref_xz_symbolic is not None:
            self.validate_ref_xz_symbolic_format(self.ref_xz_symbolic)

            # original_xs, original_zs = self.grid2d.shape
            # center_val = [original_xs*self.dl/2, original_zs*self.dl/2]
            center_val = self.center_in_unit_length

            ref_xz_in_unit_length = []
            for ref_xz_each, center_val in zip(self.ref_xz_symbolic, center_val):
                if ref_xz_each in ['o','O','0',0]:
                    ref_xz_in_unit_length.append(0.0)
                else:
                    ref_xz_in_unit_length.append(float(center_val))
            
            ref_val_flag = True
            
        if self.ref_xz_override is not None:
            v = self.ref_xz_override

            if ref_val_flag:
                raise ValueError("Either ref_xz_symbolic or ref_xz_override must be None. Provided: Both are not None.")
            
            self.validate_ref_xz_override(self.ref_xz_override)           
            
            ref_xz_in_unit_length = v
            ref_val_flag = True
        
        if not ref_val_flag:
            raise ValueError("Either ref_xz_symbolic or ref_xz_override must be None. Provided: Both are None.")
            
        return np.array(ref_xz_in_unit_length)
    
    @property
    def get_config(self):
        """
        Return a serializable configuration of the obstruction.

        Returns
        -------
        dict
            Dictionary containing obstruction configuration.
        """  
        self_config = {}
        self_config['center_in_unit_length'] = self.center_in_unit_length
        self_config['dl'] = self.dl
        self_config['grid2d'] = self.grid2d
        self_config['ref_xz_override'] = self.ref_xz_override
        self_config['ref_xz_symbolic'] = self.ref_xz_symbolic
        self_config['description'] = self.description
        self_config['shape'] = self.shape
        self_config['snap_to_dl'] = self.snap_to_dl
        return self_config

    @classmethod
    def from_config(cls, config_dict):
        """
        Reconstruct an obstruction object from configuration data.

        Parameters
        ----------
        config_dict : dict
            Configuration dictionary produced by `get_config`.

        Raises
        ------
        ValueError
            If the configuration dictionary is invalid.
        """
        if not isinstance(config_dict, dict):
            raise TypeError("Expected a dictionary.")
        try:
            obj = cls.__new__(cls) 
            obj.center_in_unit_length = config_dict['center_in_unit_length']
            obj.dl = config_dict['dl']
            obj.grid2d = config_dict['grid2d']
            obj.ref_xz_override = config_dict['ref_xz_override']
            obj.ref_xz_symbolic = config_dict['ref_xz_symbolic']
            obj.description = config_dict['description']
            obj.shape = config_dict['shape']
            obj.snap_to_dl = config_dict['snap_to_dl']
            return obj

        except (KeyError, TypeError) as e:
            raise ValueError(f"Invalid config dictionary: {e}")    
    
class _Obstruction2DShapeFunctions(_Obstruction2DFunctions):   
    """
    Shape manipulation utilities for 2D obstructions.

    Extends base obstruction utilities with geometric transformations
    and spatial queries.
    """
    def __init__(self, obs_grid_dl:float, ref_xz_symbolic = ['c', 'c'], snap_to_dl:bool=True):
        super().__init__(obs_grid_dl, ref_xz_symbolic, snap_to_dl)
                
    def shift_grid_one_axis(self, shift_axis='x', shift_val_in_length_unit=0, allow_negative_shift = False):
        """
        Shift the obstruction grid along one axis.

        Parameters
        ----------
        shift_axis : {'x', 'z'}, optional
            Axis along which to shift.
        shift_val_in_unit_length : float, optional
            Shift distance in physical units.
        allow_negative_shift : bool, optional
            If False, negative shifts raise an error.
        """
        assert self.shape is True, "Obstacles class is not defined properly"
        assert shift_axis in ['x', 'z'], f"shift axis can only be either 'x' or 'z'. Not {shift_axis}"

        # Auto snapping
        shift_in_grid = int(np.round(shift_val_in_length_unit/self.dl,0))
        shift_val_in_length_unit = shift_in_grid * self.dl
        
        if shift_in_grid!=0:
            if not allow_negative_shift: 
                if shift_in_grid < 0:
                    raise ValueError(f"Shift values must be non-negative. Provided {shift_in_grid}")
    
            # Get original grid dimensions
            original_xs, original_zs = self.grid2d.shape
        
            # Calculate new grid dimensions
            if shift_axis == 'x':
                new_xs = original_xs + shift_in_grid
                new_zs = original_zs
            else:
                new_xs = original_xs
                new_zs = original_zs + shift_in_grid
        
            # Create a new grid filled with zeros
            new_grid = np.zeros((new_xs, new_zs), dtype=int)
        
            # Place the original grid in the new grid at the specified shift
            if shift_in_grid>0:
                new_grid[new_xs-original_xs:new_xs, new_zs-original_zs:new_zs] = self.grid2d
            else:
                if shift_axis == 'z':
                    new_grid[:, 0:new_zs] = self.grid2d[:, -shift_in_grid:]
                else:
                    new_grid[0:new_xs, :] = self.grid2d[-shift_in_grid:, :]
        
            self.grid2d = new_grid
            self.center_in_unit_length = [self.center_in_unit_length[0] + (new_xs-original_xs)*self.dl, self.center_in_unit_length[1] + (new_zs-original_zs)*self.dl]
            
            if self.ref_xz_override is not None:
                self.ref_xz_override = [self.ref_xz_override[0] + (new_xs-original_xs)*self.dl, self.ref_xz_override[1] + (new_zs-original_zs)*self.dl]
            self.description += f', then shifted in {shift_axis}-axis by {shift_val_in_length_unit}'

    def scale_shapes(self, scale_factor):
        """
        Scale the obstruction geometry.

        Parameters
        ----------
        scale_factor : float
            Positive scaling factor.
        """
        assert self.shape is True, "Obstacles class is not defined properly"
        
        if scale_factor <= 0:
            raise ValueError("Scale factor must be a positive number.")

        new_x_len = int(np.round(self.grid2d.shape[0] * scale_factor,0))
        new_z_len = int(np.round(self.grid2d.shape[1] * scale_factor,0))

        scaled_grid = np.zeros((new_x_len, new_z_len), dtype=int)
        
        scaled_grid = f.remeshing_2D_matrix(x_old = np.arange(self.grid2d.shape[0])*self.dl+self.dl/2,
                                            x_new = np.arange(new_x_len)*self.dl/scale_factor+self.dl/scale_factor/2,
                                            z_old = np.arange(self.grid2d.shape[1])*self.dl+self.dl/2,
                                            z_new = np.arange(new_z_len)*self.dl/scale_factor+self.dl/scale_factor/2,
                                            matrix_2d = self.grid2d, interp_method = 'nearest')
        # Update the grid and resolution
        self.grid2d = scaled_grid
        self.center_in_unit_length = [i*scale_factor for i in self.center_in_unit_length] # To check if it is okay for non-integer scaling
        if self.ref_xz_override is not None:
            self.ref_xz_override = [i*scale_factor for i in self.ref_xz_override] # To check if it is okay for non-integer scaling
        self.description += f', then scaled by factor of {scale_factor}'
    
    def expand_grid(self, new_grid_xlen, new_grid_zlen, warn_truncate=True):
        """
        Expand or truncate the grid to new dimensions.

        Parameters
        ----------
        new_grid_xlen : int
            New grid size in x-direction.
        new_grid_zlen : int
            New grid size in z-direction.
        """
        assert self.shape is True
        if new_grid_xlen is None:
            new_grid_xlen = self.grid2d.shape[0]
        if new_grid_zlen is None:
            new_grid_zlen = self.grid2d.shape[1]
        new_grid_xlen = f.check_integer(new_grid_xlen)
        new_grid_zlen = f.check_integer(new_grid_zlen)

        self_xlen, self_zlen = self.grid2d.shape
        if warn_truncate:
            if not (self_xlen<=new_grid_xlen and self_zlen<=new_grid_zlen):
                warnings.warn(f"WARNING: shape (format: x * z) of old grid {self_xlen} x {self_zlen} is greater than {new_grid_xlen} x {new_grid_zlen}. Obstacles might get removed from the model")
                    
        new_grid = np.zeros((max(self_xlen, new_grid_xlen), max(self_zlen, new_grid_zlen)), dtype = int)
        new_grid[:self_xlen, :self_zlen] = self.grid2d
        new_grid = new_grid[:new_grid_xlen, :new_grid_zlen]
        self.grid2d = new_grid
        
    def merge_shapes(self, obstruction2d_instance_other:"_Obstruction2DShapeFunctions"):
        """
        Merge another obstruction into this one.

        Parameters
        ----------
        obstruction2d_instance_other : _Obstruction2DShapeFunctions
            Obstruction instance to merge.
        """
            
        obstruction2d_instance_other = copy.deepcopy(obstruction2d_instance_other)
        assert self.shape is True, "Obstruction2D class is not defined properly"
        assert obstruction2d_instance_other.shape is True, "obstruction2d_instance_other is not defined properly"
        assert self.dl == obstruction2d_instance_other.dl, "Dimension Error: Merged obstruction must have same dl (grid spacings)"
        
        self_xlen, self_zlen = self.grid2d.shape
        util2_xlen, util2_zlen = obstruction2d_instance_other.grid2d.shape
        # print(shift)
        # print(self_xlen, util2_xlen, self_zlen, util2_zlen)
        new_grid_xlen, new_grid_zlen = np.max([self_xlen, util2_xlen]), np.max([self_zlen, util2_zlen])

        self.expand_grid(new_grid_xlen, new_grid_zlen)
        
        # Old style replacing 0s
        # obstruction2d_instance_other.expand_grid(new_grid_xlen, new_grid_zlen)

        # # Initialize the merged array
        # merged = np.zeros_like(self.grid2d)
    
        # mask_a_nonzero = (self.grid2d != 0)
        # mask_b_nonzero = (obstruction2d_instance_other.grid2d != 0)
    
        # # Apply the merging rules using numpy's where function
        # merged = np.where(mask_b_nonzero, obstruction2d_instance_other.grid2d, merged)  # b != 0
        # merged = np.where(mask_a_nonzero, self.grid2d, merged)  # a != 0 (replace it with a even if merged already have values, i.e priortize b)
    
    
        # New style just merge the origins. Use .shift_grid if needed.
        self.grid2d[:util2_xlen, :util2_zlen] = obstruction2d_instance_other.grid2d
        
        self.description+= f', then merged with Utils2d of {obstruction2d_instance_other.description}' 
            
    def clear_utils_properties(self, dl=None, ref_xz_symbolic = ['c', 'c'], snap_to_dl=True):
        """
        clear the inputs for Utils_2D object for redefining.
        """
        if dl is None:
            self.__init__(self.dl, ref_xz_symbolic, snap_to_dl)
        else:
            self.__init__(dl, ref_xz_symbolic, snap_to_dl)
            
    def query_points_in_obstruction(self, actual_point_coord, shift_ref2d_to_act=[0,0]):
        """
        Query whether multiple points in point_coord (x,z) lie inside the  grid_2d
        using nearest interpolation. Optimized by cropping the grid to the
        minimal bounding box that covers both the obstruction and the points.

        Note: obs_point_coord = actual_point_coord - shift_ref2d_to_act + ref2d.
        That obs_point_coord will be checked if lies in the obstruction2D grid.
        
        Parameters
        ----------
        actual_point_coord : numpy.ndarray, shape (N, 2)
            Point coordinates in physical units.
        shift_ref2d_to_act : array-like of length 2, optional
            Shift from reference coordinates to actual coordinates.

        Returns
        -------
        numpy.ndarray, shape (N,)
            Integer values indicating obstruction membership
            (0 = outside, non-zero = inside).
        """
        assert self.shape is True, "Obstruction2D class is not defined properly"
        grid_2d = self.grid2d
        grid_2d = np.pad(grid_2d, pad_width=1, mode='constant', constant_values=0) #Note padded one so ref_coord also add 1 

        center_2d = np.asarray(self.get_ref_xz_in_unit_length())
        shift_ref2d_to_act = np.asarray(shift_ref2d_to_act)
        actual_point_coord = np.asarray(actual_point_coord)

        if shift_ref2d_to_act.shape != (2,):
            raise ValueError("shift_ref2d_to_act must have shape (2,)")

        if actual_point_coord.ndim != 2 or actual_point_coord.shape[1] != 2:
            raise ValueError("actual_point_coord must have shape (N,2)")

        if center_2d.shape != (2,):
            raise ValueError("center_2d must have shape (2,)")

        # --- Numeric checks ---
        if not np.issubdtype(shift_ref2d_to_act.dtype, np.number):
            raise TypeError("shift_ref2d_to_act must contain numeric values")

        if not np.issubdtype(actual_point_coord.dtype, np.number):
            raise TypeError("actual_point_coord must contain numeric values")

        if not np.issubdtype(center_2d.dtype, np.number):
            raise TypeError("center_2d must contain numeric values")

        n_x, n_z = grid_2d.shape
        max_x, max_z = (n_x) * self.dl, (n_z) * self.dl

        x_grid =  np.arange(n_x)*self.dl+self.dl/2 - self.dl # 1 grid point due to padding
        z_grid =  np.arange(n_z)*self.dl+self.dl/2 - self.dl

        actual_point_coord = actual_point_coord - shift_ref2d_to_act + center_2d
        x, z = actual_point_coord[:,0], actual_point_coord[:,1]

        # Step 1: Cheap filters
        keep = (-self.dl <= x) & (x <= max_x) & (-self.dl <= z) & (z <= max_z)

        vals = np.zeros(len(actual_point_coord), dtype=int)

        if not np.any(keep):
            return vals  # all rejected

        # Step 2: Cropping indices (optional but keeps interpolation efficient)
        n_extra_grids = 2
        min_z_idx = max(0, int(np.floor((z[keep].min() - z_grid.min()) / self.dl)) - n_extra_grids)
        max_z_idx = min(n_z-1, int(np.ceil((z[keep].max() - z_grid.min()) / self.dl)) + n_extra_grids)
        min_x_idx = max(0, int(np.floor((x[keep].min() - x_grid.min()) / self.dl)) - n_extra_grids)
        max_x_idx = min(n_x-1, int(np.ceil((x[keep].max() - x_grid.min()) / self.dl)) + n_extra_grids)

        cropped_grid = grid_2d[min_x_idx:max_x_idx + 1, min_z_idx:max_z_idx + 1]
        cropped_z = z_grid[min_z_idx:max_z_idx + 1]
        cropped_x = x_grid[min_x_idx:max_x_idx + 1]

        interp = RegularGridInterpolator(
            (cropped_x, cropped_z),
            cropped_grid,
            method="nearest",
            bounds_error=False,
            fill_value=0,
        )

        vals[keep] = interp(np.column_stack((x[keep], z[keep]))).astype(int)
        return vals

class Obstruction2D(_Obstruction2DShapeFunctions):    
    """
    Concrete 2D obstruction geometry class.

    Supports basic geometric shapes such as circles and rectangles.
    """
    def __init__(self, dl:float, ref_xz_symbolic = ['c', 'c'], snap_to_dl:bool=True):
        super().__init__(dl, ref_xz_symbolic, snap_to_dl)
                
    def circle_2d(self, d, obstruction_id = 1, warn_adjustments=False):
        """
        Create a circular 2D obstruction.

        Parameters
        ----------
        d : float
            Diameter of the circle.
        obstruction_id : int, optional
            Integer identifier assigned to the obstruction.
        warn_adjustments : bool, optional
            If True, warn when dimensions are adjusted for discretization.
        """
        assert self.shape is False, "ERROR: utils shape has already been defined"
        if d<0:
            raise ValueError("Diameter of the circle must be zero/positive")
        
        obstruction_id = f.check_obstruction_id(obstruction_id)
        
        dl = self.dl
        d_adj = d
        if self.snap_to_dl:
            d_adj = int(np.round(d/dl,0))*dl
            
            if warn_adjustments and not f.is_close(d_adj, d):
                warnings.warn(f"Adjusted radius of circle is {d_adj}, changed from {d}")
        
        # Determine the number of grid points based on the radius and grid resolution (del_x, del_z)
        n_utilsgrid_x = int(np.round(d_adj / self.dl,0))
        n_utilsgrid_z = int(np.round(d_adj / self.dl,0))
        
        # Create a blank 2D array (grid) of zeros
        grid = np.zeros((n_utilsgrid_x, n_utilsgrid_z), dtype=int)
        r = d_adj/2
        x = np.arange(n_utilsgrid_x) * self.dl + self.dl/2
        z = np.arange(n_utilsgrid_z) * self.dl + self.dl/2
        xx, zz = np.meshgrid(x, z, indexing='ij')
        dist2 = (xx - r)**2 + (zz - r)**2
        grid[dist2 <= r*r] = obstruction_id
        
        # Update object attributes
        self.grid2d = grid
        self.description = f'Circular 2d of diameter {d_adj}'
        self.shape = True
        self.center_in_unit_length = [d_adj/2, d_adj/2]
        
    def rectangle_2d(self, lx, lz, obstruction_id=1, warn_adjustments=False):
        """
        Create a rectangular 2D obstruction.

        Parameters
        ----------
        lx : float
            Length in the x-direction.
        lz : float
            Length in the z-direction.
        obstruction_id : int, optional
            Integer identifier assigned to the obstruction.
        warn_adjustments : bool, optional
            If True, warn when dimensions are adjusted for discretization.
        """
        assert self.shape is False, "ERROR: utils shape has already been defined"
        if lx < 0 or lz < 0:
            raise ValueError(f"Dimensions lx, lz must be positive/zero. Provided {lx} and {lz}")
        obstruction_id = f.check_obstruction_id(obstruction_id)
        
        dl = self.dl
        lx_adj = lx
        lz_adj = lz
        
        #Note: For rectangles... snap_to_dl True and False gives same result.
        if self.snap_to_dl:
            ## Adjustment for better discretization
            lx_adj = int(np.round(lx/dl,0))*dl
            lz_adj = int(np.round(lz/dl,0))*dl

        # else:
        #     n_grid_x = 1
        #     n_utilsgrid_x = 1
        #     mid_grid_x = 0 # Or 1?

        if warn_adjustments:
            if not f.is_close(lx_adj, lx) or not f.is_close(lz_adj, lz):
                warnings.warn(f"Adjusted size of rectangle is {lx_adj} x {lz_adj}, changed from {lx} x {lz}")
            
        # Determine the number of grid points based on the radius and grid resolution (del_x, del_z)
        n_utilsgrid_x = int(np.round(lx_adj / self.dl,0))
        n_utilsgrid_z = int(np.round(lz_adj / self.dl,0))
        
        self.grid2d = np.full((n_utilsgrid_x, n_utilsgrid_z), obstruction_id, dtype=int)
        self.description = f'Rectangular 2d of size (lx x lz) = ({lx_adj:.6g} x {lz_adj:.6g})'
        self.shape = True
        self.center_in_unit_length = [lx_adj/2, lz_adj/2]
        