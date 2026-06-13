# This file is part of modgen2d a Python package for ...
# Copyright (C) XXXX Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
# LICENSE

"""
Define a two-dimensional lithological domain class.

This module provides a read-only class representing a 2D lithological domain 
(matrix of layer IDs) with optional groundwater table (GWT) depth, utilities,
and interfaces. It also provides plotting and remeshing capabilities.
"""
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import modgen2d.general_functions as f
import copy,warnings

from modgen2d.discretized_domain2d import DiscretizedDomain2D
from modgen2d.obstruction2d import Obstruction2D
from modgen2d.interface.global_soil_interface_config import GlobalSoilInterfaceConfig
from modgen2d._plots import _plot_lit_domain

class LithologicalDomain2DReadOnly():
    """
    Class representing a read-only 2D lithological domain with layer IDs.
    
    Parameters
    ----------
    domain : DiscretizedDomain2D
        Domain in which the lithological interfaces are defined.
    name : str
        Name of the lithological domain.
    """
    def __init__(self, domain:DiscretizedDomain2D, name: str):
        """
        Initialize a 'LithologicalDomain2DReadOnly' instance.
        """
        self.domain = domain
        self.name = name
        self.lm_type = 'NA'
        self._lithological_matrix = None
        self.interface_config_revision_id = GlobalSoilInterfaceConfig.get_revision_id()
        
        ##TODO add unittests
        self.lit_ids_expected = []  
        
        #For lithologicalDomain from Interface
        self.gwt_depth = None
        
        #For Lithological Domain from Obstruction2D
        self.obstruction2d_dict_list = []        
        self.obstruction_overlap = None
        
        self.merged_lit = False
        self.init_domain = None #None means domain has never been changed.
        self.lit_order = None
        
    @property
    def lithological_matrix(self):
        """
        numpy.ndarray
            Two-dimensional matrix of lithological identifiers.

        Each entry represents the lithological ID assigned to the
        corresponding cell in the discretized domain.
        """
        return self._lithological_matrix

    @lithological_matrix.setter
    def lithological_matrix(self, value):
        """Set the lithological matrix and validate its contents."""
        self._lithological_matrix = value
        self._validate_lithological_matrix()
        self.check_shape()
        
    @staticmethod
    def _validate_lit_ids(id_list):
        # Must be a list
        if not isinstance(id_list, list):
            raise TypeError("lit_ids must be provided as a list of strings.")
        
        # Must be unique
        if len(id_list) != len(set(id_list)):
            raise ValueError("lit_ids must contain unique values.")

        for s in id_list:
            if not isinstance(s, str):
                raise TypeError(
                    f"lit_ids must be a 1D list of strings. "
                    f"Lit_id {s} is of type {type(s).__name__}."
                )
                
            # Allowed placeholder
            if s == "X":
                continue

            # Pure integer label
            if s.isdigit():
                continue

            # feature_id_<int>
            parts = s.split("_", 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid lithological entry: '{s}'")

            prefix, suffix = parts
            if not prefix or not suffix.isdigit():
                raise ValueError(f"Invalid lithological entry: '{s}'")
    
    def _set_lit_ids_expected(self, lit_ids_expected, merge=False):
        self._validate_lit_ids(lit_ids_expected)
        expected_vals = set(lit_ids_expected)
        
        if merge:
            expected_vals = expected_vals | set(self.lit_ids_expected)
        
        self.lit_ids_expected = sorted(expected_vals)
    
    def _validate_lithological_matrix(self):
        """
        Validate the lithological matrix for allowed values.

        Checks:
        - Must be a numpy array or None.
        - Cannot contain NaN or None.
        - Entries must be integers, 'X', or strings like 'prefix_<int>'.

        Raises
        ------
        TypeError
            If lithological_matrix is not a numpy array.
        ValueError
            If invalid entries are detected.
        """
        self._validate_lit_ids(self.lit_ids_expected)
        
        lithological_matrix = self.lithological_matrix
        if lithological_matrix is None:
            return True

        if not isinstance(lithological_matrix, np.ndarray):
            raise TypeError("lithological_matrix must be a numpy array or None.")

        # Explicit NaN / None check
        if np.any(pd.isna(lithological_matrix)):
            raise ValueError("lithological_matrix contains NaN or None values.")

        # Work only on unique string values (small loop)
        unique_vals = np.unique(lithological_matrix.astype(str))

        # Expected lithology IDs (assumed iterable of strings)
        expected_vals = set(self.lit_ids_expected)

        # Find unexpected values
        unexpected = set(unique_vals) - expected_vals

        if unexpected:
            raise ValueError(
                f"Unexpected lithology IDs found in lithological_matrix: {sorted(unexpected)}. "
                f"Expected only: {sorted(expected_vals)}."
            )

        # print("Ran auto test: Check")
        return True
    
    @staticmethod
    def check_for_Xs(lit_matrix_value):
        """
        Check whether a lithological matrix contains unresolved placeholders.

        Parameters
        ----------
        lit_matrix_value : array-like
            Lithological matrix to inspect.

        Returns
        -------
        bool
            True if the matrix contains the placeholder value ``"X"``.
        """
        return "X" in np.unique(np.asarray(lit_matrix_value, dtype=str))

    def print(self):
        """
        Print a summary of the lithological domain.
        """
        print(f"N_x_coord = {self.lithological_matrix.shape[0]}, N_z_coord = {self.lithological_matrix.shape[1]}")
        print(f"Expected lit_ids: {self.lit_ids_expected}")
        print("Layered Matrix : \n", self.lithological_matrix.T) 
    
    def get_feature_id_and_lit_val(self):
        """
        Group lithological identifiers by feature type.

        Returns
        -------
        dict
            Dictionary where keys are feature prefixes and values are
            lists of associated lithological indices.

        Examples
        --------
        ``["0", "1", "U_1", "U_2"]`` becomes::

            {
                "def": [0, 1],
                "U": [1, 2]
            }
        """
        self._validate_lithological_matrix()
        
        if not self.lit_ids_expected:
            return {}

        # Use only unique entries (much faster and avoids duplicates)
        arr = np.unique(self.lit_ids_expected)
        arr = arr[arr != 'X']
        
        # mask for pure digits
        mask_def = np.char.isdigit(arr)

        # Everything else is candidate for prefix-number
        non_def = arr[~mask_def]

        # Partition by underscore
        sep = '_'
        result = {"def": arr[mask_def].astype(int).tolist()}
        for s in non_def:
            parts = s.split(sep, 1)

            # Case 1: no underscore → prefix only
            if len(parts) == 1:
                prefix = parts[0]
                suffix = None
            else:
                prefix, suffix = parts

                # ERROR: suffix must be numeric
                if not suffix.isdigit():
                    raise ValueError(
                        f"Invalid numeric suffix in: {s}"
                    )

                suffix = int(suffix)

            result.setdefault(prefix, [])
            if suffix is not None:
                result[prefix].append(suffix)
        return result
    
    def check_shape(self):
        """Ensure lithological matrix matches the domain shape."""
        domain_shape = self.domain.shape
        lit_shape = self.lithological_matrix.shape
        if domain_shape != lit_shape:
            raise ValueError(f"Matrix shape mismatch. Domain shape {domain_shape} != lit_shape {lit_shape}.")
        
    
    def plot(self, ax=None, discrete_point_size=0, white_edges_size=0, plot_gwt=True, water_alpha = 0.4, gwt_kw={}, legend=True, try_clean_legend=False,
               id2material_dict = None, title='Lithological Domain',
               plot_interfaces = False, plot_interfaces_kw={},
               color_map = {
                        'def': plt.get_cmap('tab20', 10),      # For integer values
                        'U': plt.get_cmap('Set3', 10)   # For "U-{x}" values
                }):
        """
        Plot the lithological domain.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes object to draw on. Creates a new figure if None.
        discrete_point_size : float, optional
            Size of discrete scatter points; 0 disables scatter.
        white_edges_size : float, default 0
            Size of white edges of pixels.
        plot_gwt : bool, default True
            Plot groundwater table.
        gwt_kw : dict,
            keywords for controlling gwt_plot
        
        legend : bool, optional
            Whether to display the legend.
        id2material_dict : dict, optional
            Maps IDs to material names.
        title : str, optional
            Plot title.
        plot_interfaces : bool, optional
            Whether to overlay soil interfaces.
        color_map : dict, optional
            Prefix-to-colormap dictionary.
        """
        if ax is None:
            fig,ax = plt.subplots()

        # if id2material_dict is None and try_clean_legend:
        #     feature_id_dict = self.get_feature_id_and_lit_val()
            
        
        ax = _plot_lit_domain(self.domain, self.lithological_matrix, self.gwt_depth, ax=ax, 
                        discrete_point_size=discrete_point_size, white_edges_size=white_edges_size, 
                        plot_gwt=plot_gwt, water_alpha=water_alpha, gwt_kw = gwt_kw, legend=legend, try_clean_legend=try_clean_legend,
                        id2material_dict = id2material_dict, title=title,
                        color_map = color_map)
            
        # Plot Boundary:
        if plot_interfaces:
            if 'linestyle' not in plot_interfaces_kw.keys():
                plot_interfaces_kw['linestyle'] = '--'
            if 'color' not in plot_interfaces_kw.keys():
                plot_interfaces_kw['color'] = 'k'
            discretizedInterfaces2D_instance = GlobalSoilInterfaceConfig.get_interface_instance()
            if discretizedInterfaces2D_instance is not None:
                n_soil_layers = discretizedInterfaces2D_instance.n_soil_layers
                for i in np.arange(n_soil_layers-1, -1, -1):
                    remesh_tech = discretizedInterfaces2D_instance.remesh_interp_method
                    if remesh_tech == 'nearest':
                        drawstyle = 'steps-mid'
                    elif remesh_tech == 'linear':
                        drawstyle = 'default'
                        if self.init_domain is not None:
                            warnings.warn("Looks like the lit domain has been remeshed with linear after creation.")
                    else:
                        drawstyle = 'steps-mid'
                        warnings.warn(f"Interfaces might not reflect the exact interpolation in the plots except for 'linear' and 'nearest'. Provided {remesh_tech}.")
                    ax.plot(discretizedInterfaces2D_instance.domain.get_interface_x_centers,
                            discretizedInterfaces2D_instance.interfaces_matrix[:, i],
                            drawstyle=drawstyle,
                            **plot_interfaces_kw,
                        )
        
        return ax

    def remeshing_lithological_matrix(self, new_dx, new_dz, interp_method='nearest', replace=True):
        """
        Remesh (coarsen or refine) the lithological matrix based on new grid spacing.

        This method interpolates the lithological matrix onto a new discretized domain
        with spacings `new_dx` and `new_dz`. Two interpolation methods are supported:
        'nearest' and 'nearest_up'. Optionally, the remeshed matrix can replace the
        existing one or return a copy.

        Parameters
        ----------
        new_dx : float
            Desired spacing in the X-direction for the remeshed domain.
        new_dz : float
            Desired spacing in the Z-direction for the remeshed domain.
        interp_method : str, optional
            Interpolation method to use for remeshing. Options:

            - 'nearest' : Assigns the value of the nearest original grid cell.
            - 'nearest_up' : Similar to 'nearest' but may favor higher layer IDs, useful when upscaling layered properties.

            Default is 'nearest'.
        replace : bool, optional
            If True, the remeshed matrix will replace the current lithological matrix
            and domain in-place. If False, a new `LithologicalDomain2DReadOnly` object
            with the remeshed matrix is returned. Default is True.

        Returns
        -------
        LithologicalDomain2DReadOnly or None
            - If `replace=False`, returns a new instance of `LithologicalDomain2DReadOnly` with the remeshed matrix.
            - If `replace=True`, updates the current object in-place and returns None.

        Raises
        ------
        ValueError
            If an unsupported interpolation method is provided.

        Notes
        -----
        - The original domain is stored in `init_domain` if it has never been modified.
        - The method preserves unique lithology identifiers, including integers and prefixed values like 'U_<int>'.
        - Colors, plotting, and validation remain compatible after remeshing.
        """
        self_copy = copy.deepcopy(self)  #Makes sure the change in the object is local to this function only.
        org_dx, org_dz = self.domain.dhs
        if org_dx != new_dx or org_dz == new_dz:

            if interp_method!='nearest' and interp_method!='nearest_up':
                raise ValueError(f"interp_method cannot be other than 'nearest' or 'nearest_up' for replacement. Provided {interp_method}")
                
            new_domain = self.domain.remesh(new_dx, new_dz)
            
            unique_values, int_mapp = np.unique(self.lithological_matrix, return_inverse=True)
            int_mapp = int_mapp.reshape(self.lithological_matrix.shape)
            int_values = np.arange(len(unique_values))
            
            remeshed_int_mapp = f.remeshing_2D_matrix(x_old = self.domain.x_centers, 
                                                      z_old = self.domain.z_centers, 
                                                      x_new = new_domain.x_centers,
                                                      z_new = new_domain.z_centers,
                                                      matrix_2d=int_mapp, interp_method=interp_method)
            
            self_copy.domain = new_domain
            self_copy.lithological_matrix = unique_values[remeshed_int_mapp.astype(int)]
            # self_copy._validate_lithological_matrix() #Should be done automatically with setter
            
            self_copy.lm_type = f'{self_copy.lm_type}_remeshed'
            if self_copy.init_domain is None:
                self_copy.init_domain = self.domain

        if replace:  #to verify
            # Update `self`'s attributes in-place to reflect `self_copy`
            self.__dict__.update(self_copy.__dict__)
        else:
            return self_copy
            
    @property
    def get_config(self):
        """
        Return a serializable dictionary representing the lithological domain.
        """
        self_config = {}
        self_config['domain'] = self.domain.get_config
        self_config['gwt_depth'] = self.gwt_depth
        if self.init_domain is None:
            self_config['init_domain'] = self.init_domain
        else:
            self_config['init_domain'] = self.init_domain.get_config
        self_config['interface_config_revision_id'] = self.interface_config_revision_id
        
        self_config['lithological_matrix'] = self.lithological_matrix
        self_config['lm_type'] = self.lm_type
        self_config['merged_lit'] = self.merged_lit
        self_config['name'] = self.name
        self_config['lit_order'] = self.lit_order
        self_config['lit_ids_expected'] = self.lit_ids_expected
       
        self_config['obstruction2d_dict_list'] = {}
        if len(self.obstruction2d_dict_list)!=0:
            for i,obstruction2D_dict in enumerate(self.obstruction2d_dict_list):
                self_config['obstruction2d_dict_list'][f'{i}'] = {
                    'obstruction_inst': obstruction2D_dict['obstruction_inst'].get_config,
                    'shift_ref2d_to_xy': obstruction2D_dict['shift_ref2d_to_xy'],        
                    'feature_id': obstruction2D_dict['feature_id'],   
                    'y_shift_for_surface_adj': obstruction2D_dict['y_shift_for_surface_adj'],
                }
       
        # Only for Lithological Domain from Obstuction3D
        self_config['obstruction_overlap'] = self.obstruction_overlap
        return self_config

    @classmethod
    def from_config(cls, config_dict):
        """
        Create a LithologicalDomain2DReadOnly instance from a configuration dictionary.

        Parameters
        ----------
        config_dict : dict
            Dictionary produced by `get_config`.

        Returns
        -------
        LithologicalDomain2DReadOnly
            Reconstructed instance.
        """
        if not isinstance(config_dict, dict):
            raise TypeError("Expected a dictionary.")
        
        try:
            obj = cls.__new__(cls)
            discretizedDomain2D = DiscretizedDomain2D.from_config(config_dict['domain'])
            name = config_dict['name']
            obj.domain = discretizedDomain2D
            obj.name = name
            
            # if config_dict['gwt_depth'] == 'None':
            #     config_dict['gwt_depth'] = None
            obj.gwt_depth = config_dict['gwt_depth']
            
            if config_dict['init_domain'] is None:
                obj.init_domain = None
            else:
                obj.init_domain = DiscretizedDomain2D.from_config(config_dict['init_domain'])

            obj.interface_config_revision_id = config_dict['interface_config_revision_id']
            obj.lm_type = config_dict['lm_type']
            obj.merged_lit = config_dict['merged_lit']
            obj.obstruction_overlap = config_dict['obstruction_overlap']
            obj.lit_order = config_dict['lit_order']
            obj.lit_ids_expected = config_dict['lit_ids_expected']
            obj.lithological_matrix = config_dict['lithological_matrix']
            
            obj.obstruction2d_dict_list = []
            obstruction2D_instance_list = config_dict['obstruction2d_dict_list']
            if len(obstruction2D_instance_list)!=0:
                for _,obstruction2D_dict_raw in obstruction2D_instance_list.items():
                    obstruction2D_dict = {
                        'obstruction_inst': Obstruction2D.from_config(obstruction2D_dict_raw['obstruction_inst']),
                        'shift_ref2d_to_xy': obstruction2D_dict_raw['shift_ref2d_to_xy'],        
                        'feature_id': obstruction2D_dict_raw['feature_id'],   
                        'y_shift_for_surface_adj': obstruction2D_dict_raw['y_shift_for_surface_adj'],
                    }
                    obj.obstruction2d_dict_list.append(obstruction2D_dict)
            return obj

        except (KeyError, TypeError) as e:
            raise ValueError(f"Invalid config dictionary: {e}")
        
    # def __eq__(self, other):
    #     if not isinstance(other, LithologicalDomain2DReadOnly):
    #         return NotImplemented
        
        # units_check = self.units_config == other.units_config
        # spans_check = np.allclose(self._spans_in_domain_len_units, other._spans_in_domain_len_units)
        # dhs_check = np.allclose(self._dhs_in_domain_len_units, other._dhs_in_domain_len_units)
        # return (
        #     units_check
        #     and spans_check
        #     and dhs_check
        # )

