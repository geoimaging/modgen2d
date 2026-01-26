"""
Modules for each 2D lithological domain and its associated simulated property fields. 
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
# from IPython.display import clear_output
import numpy as np
import geomodgen2d.general_functions as f
from geomodgen2d.lithological_domain2d import LithologicalDomain2D, LithologicalDomain2DFromObstruction2D, LithologicalDomain2DReadOnly
import warnings

class GeneratedModel2D:
    """
    Represents a 2D lithological domain with associated simulated property profiles.
    """
    def __init__(self, lithological_domain_instance:LithologicalDomain2D, gwt_depth, lit_id2material_dict, simulated_val_for_ignored_lit_property=-99999):
        """
        Initializes 'GeneratedModel2D' object instance.

        Parameters
        ----------
        lithological_domain_instance : LithologicalDomain2D
            Instance of a 2D lithological domain.
        gwt_depth : float
            Depth of the groundwater table for wet/dry classification.
        lit_id2material_dict : dict
            Dictionary mapping lithological IDs (str) to arrays of material properties.
        simulated_val_for_ignored_lit_property : int, default=-99999
            Value used for ignored lithological IDs during simulation.
        """
        self.lit_domain = lithological_domain_instance
        self.lit_order = lithological_domain_instance.lit_order
        self.gwt_depth = gwt_depth
        self.lit_id2material_dict = {}
        self.simulated_profiles = {}
        self._locked = False
        self._simulated_val_for_ignored_lit_property = simulated_val_for_ignored_lit_property #Make sure consistent with all generated profiles and spatial_simulator
        lithological_domain_instance.check_shape()
        
        unique_ids = np.unique(self.lit_domain.lithological_matrix.astype(str))
        unique_ids = unique_ids[unique_ids!="X"]
        
        # Check that all unique IDs are keys in lit_id2material
        missing_keys = [uid for uid in unique_ids if uid not in lit_id2material_dict]
        if missing_keys:
            raise ValueError(f"The following lithological IDs are missing in lit_id2material_dict: {missing_keys}")
        
        # Filter out any extra keys in lit_id2material_dict
        filtered_dict = {k: np.array(v) for k, v in lit_id2material_dict.items() if k in unique_ids}
        self.lit_id2material_dict = filtered_dict
    
    @property
    def simulated_val_for_ignored_lit_property(self):
        """
        int: The value used for ignored lithological IDs in simulated profiles.
        """
        return self._simulated_val_for_ignored_lit_property

    @simulated_val_for_ignored_lit_property.setter
    def simulated_val_for_ignored_lit_property(self, value):
        """
        Sets the simulated value for ignored lithologies.

        Raises
        ------
        AttributeError
            If simulated profiles already exist.
        TypeError
            If the value is not an integer.
        """
        if self.simulated_profiles is not None:
            raise AttributeError(
                "Profiles has already been generated and hence, cannot be changed."
            )
            
        if not isinstance(value, int):
            raise TypeError(
                f"simulated_value_for_ignored_lit must be an integer, got {type(value)}"
            )
            
        self._simulated_val_for_ignored_lit_property = value
            
    def check(self, ignore_lithological_ids=['X'], allow_ignored_lit_property=True):
        """
        Validates simulated profiles for shape consistency, NaN values, and ignored IDs.

        Parameters
        ----------
        ignore_lithological_ids : list of str, default ['X']
            Lithological IDs to ignore during the check.
        allow_ignored_lit_property : bool, default True
            If False, raises an error if ignored-value appears in profiles.

        Raises
        ------
        ValueError
            If shape mismatches, missing IDs, NaNs, or ignored-value inconsistencies are detected.
        """
        domain_shape = self.lit_domain.domain.shape
        lit_shape = self.lit_domain.lithological_matrix.shape
        if domain_shape != lit_shape:
            raise ValueError(f"Matrix shape mismatch. Domain shape {domain_shape} != lit_shape {lit_shape}.")
        
        unique_ids = np.unique(self.lit_domain.lithological_matrix.astype(str))
        
        # Check that all unique IDs are keys in lit_id2material
        missing_keys = (
            set(unique_ids)
            - set(self.lit_id2material_dict)
            - set(ignore_lithological_ids)
        )

        if missing_keys:
            raise ValueError(
                f"The following lithological IDs are missing in lit_id2material_dict and ignore_lithological_ids: {sorted(missing_keys)}"
            )

        #Make sure positions in numpy array of self.lit_domain.lithological_matrix wher (ignore_lithological_ids) and that of self.simualted_val_for_ignored_lit_property in self.simulated_profile is same...
        ignored_lith_mask = np.isin(self.lit_domain.lithological_matrix, ignore_lithological_ids)
        
        for key,val in self.simulated_profiles.items():
            if domain_shape != val.shape:
                raise ValueError(f"Matrix shape mismatch. Domain shape {domain_shape} != lit_shape {val.shape} for property: {key}.")
            
            contains_nan = np.isnan(val).any()
            if contains_nan!=0:
                raise ValueError(f"The simulated profile for property {key} cannot have nan-values")

            # Position of ignored values consistency check
            ignored_sim_mask = val == self.simulated_val_for_ignored_lit_property
            
            if not np.array_equal(ignored_lith_mask, ignored_sim_mask):
                raise ValueError(
                    f"Mismatch between ignored lithological ids in lit_matrix and simulated ignored-value positions in simulated profile for key {key}."
                )
                
            # Check if the profile contains any values of simulated_val_for_ignored_lit_property
            contains_negative_value = (val == self.simulated_val_for_ignored_lit_property).any()
            if contains_negative_value and not allow_ignored_lit_property:
                raise ValueError(f"Few numbers in simulated profile for property {key} is {self.simulated_val_for_ignored_lit_property}, despite not allowed with flag allow_ignored_lit_property False in the .check.")
    
    def remesh(self, new_dx, new_dz):
        """
        Resamples the lithological domain and simulated profiles onto a new grid.

        Parameters
        ----------
        new_dx : float
            New spacing in the X-direction.
        new_dz : float
            New spacing in the Z-direction.

        Returns
        -------
        GeneratedModel2D
            New model instance with remeshed domain and profiles.

        Raises
        ------
        ValueError
            If remeshed profile shape does not match the new domain.
        """
        self.lit_domain.check_shape()
        
        # Make a copy of the lithological domain
        new_lit_domain = self.lit_domain.copy()  
        new_lit_domain.remeshing_lithological_matrix(new_dx, new_dz)
        
        new_domain_shape = new_lit_domain.domain.shape
        remeshed_profiles = {}
        
        for key, val in self.simulated_profiles.items():
            remeshed_profile = f.remeshing_2D_matrix(
                x_old=self.lit_domain.domain.x_centers,
                x_new=new_lit_domain.domain.x_centers,
                z_old=self.lit_domain.domain.z_centers,
                z_new=new_lit_domain.domain.z_centers,
                matrix_2d=val,
                interp_method='nearest'
            )
            if new_domain_shape != remeshed_profile.shape:
                raise ValueError(f"Matrix shape mismatch. Domain shape {new_domain_shape} != new simulated profile shape {remeshed_profile.shape}.")
            remeshed_profiles[key] = remeshed_profile

        # Create a new GeneratedModel2D instance with remeshed domain and profiles
        new_profile_instance = GeneratedModel2D(self.lit_order, new_lit_domain, self.lit_id2material_dict)
        new_profile_instance.simulated_profiles = remeshed_profiles

        return new_profile_instance
    
    def plot_lit_domain(self, ax=None, discrete_point_size=0, legend=True,
                        use_lit_id2material_dict = True, title='Lithological Domain',
                        plot_interfaces = False,
                        color_map = {
                        'def': plt.get_cmap('tab20', 10),      # For integer values
                        'U_': plt.get_cmap('Set3', 10)   # For "U-{x}" values
                }):
        """
        Plots the lithological domain.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Matplotlib axes to plot on.
        discrete_point_size : float, default 0
            Size of scatter points representing grid centers.
        legend : bool, default True
            Display a legend.
        use_lit_id2material_dict : bool, default True
            Use the material dictionary for coloring.
        title : str, default 'Lithological Domain'
            Plot title.
        plot_interfaces : bool, default False
            Plot interfaces between layers.
        color_map : dict, optional
            Colormap for integer or utility layers.
        
        Returns
        -------
        ax : matplotlib.axes.Axes
            Axes containing the plot.
        """
        if self.lit_id2material_dict and use_lit_id2material_dict:
            lit_id2material_dict = self.lit_id2material_dict
        else:
            lit_id2material_dict = None 
        
        ax = self.lit_domain.plot(ax=ax, discrete_point_size=discrete_point_size, legend=legend,
               id2material_dict = lit_id2material_dict, title=title, plot_interfaces=plot_interfaces,
               color_map = color_map)

        return ax
    
    def plot_profile(self, main_property_name, ax=None, discrete_point_size=0, plot_gwt = True,
               vlog = False, vmin=None, vmax=None, cmap='gist_earth_r', 
               title = 'auto', legend = True, legend_label = None, legendkwargs_dict={}):
        """
        Plots a 2D property profile.

        Parameters
        ----------
        main_property_name : str
            Property name to plot.
        ax : matplotlib.axes.Axes, optional
            Matplotlib axes to plot on.
        discrete_point_size : float, default 0
            Size of scatter points.
        plot_gwt : bool, default True
            Plot groundwater table.
        vlog : bool, default False
            Apply logarithmic normalization.
        vmin, vmax : float, optional
            Color scale limits.
        cmap : str or Colormap, default 'gist_earth_r'
            Colormap.
        title : str, default 'auto'
            Plot title.
        legend : bool, default True
            Show colorbar.
        legend_label : str, optional
            Colorbar label.
        legendkwargs_dict : dict, optional
            Extra keyword arguments for colorbar.

        Returns
        -------
        ax : matplotlib.axes.Axes
            Axes containing the plot.
        vmin : float
            Minimum value used for colormap.
        vmax : float
            Maximum value used for colormap.
        """
        if ax is None:
            fig,ax = plt.subplots()

        simulated_profile_set = self.simulated_profiles
        domain = self.lit_domain.domain
        z_centers, x_centers = domain.z_centers, domain.x_centers
        span_x, span_z = domain.spans
        
        if main_property_name not in simulated_profile_set.keys():
            raise ValueError(f"main_property_name: {main_property_name} not generated yet.")
        
        data = simulated_profile_set[main_property_name].T
        
        extent=[0, span_x, span_z, 0]
        # Create a colormap from the color mapping
        if vmin is None:
            vmin = np.min(data)

        if vmax is None:
            vmax = np.max(data)
            
        if vlog:
            norm = LogNorm(vmin=vmin, vmax=vmax)
            cax = ax.imshow(data, norm=norm, cmap=cmap, extent=extent, interpolation='none') 
        else:
            cax = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, interpolation='none') 
        
        # Plot gwt
        if self.lit_domain.gwt_depth is not None and plot_gwt:
            edges_kw = dict(color='r', linestyle='dashed', linewidth=2, zorder=4000)
            ax.plot([0, span_x], [self.lit_domain.gwt_depth, self.lit_domain.gwt_depth], **edges_kw)

        x_data, z_data = np.meshgrid(x_centers, z_centers, indexing='ij')
        if discrete_point_size!=0:
            ax.scatter(x_data.flatten(), z_data.flatten(), c = 'k', s=discrete_point_size)
        
        if title=='auto':
            ax.set_title(f"Main_property_name:{main_property_name}")
        
        # Colorbar
        # if legendkwargs_dict is None:
        #     legendkwargs_dict = {
        #         'shrink':0.6,
        #         'aspect':20,
        #         'pad':0.1
        #     }
        
        if legend:
            cbar = plt.colorbar(cax, ax=ax, **legendkwargs_dict)
            cbar.set_label(legend_label)
        
        ax.axis('scaled')
        ax.set(
            xlim= [0, span_x],
            ylim= [span_z, 0],
            xlabel='X',
            ylabel='Z',
        )

        return ax, vmin, vmax
    
    @property
    def get_config(self):
        """
        Export configuration.

        Returns
        -------
        dict
            Serializable configuration.
        """
        self_config = {}
        self_config['properties_metadata'] = {}
        self_config['properties_metadata']['gwt_depth'] = self.gwt_depth
        if self.lit_domain is None:
            self_config['lit_domain'] = self.lit_domain
        else:
            self_config['lit_domain'] = self.lit_domain.get_config
        self_config['properties_metadata']['lit_id2material_dict'] = self.lit_id2material_dict
        self_config['properties_metadata']['lit_order'] = self.lit_order
        self_config['simulated_profiles'] = self.simulated_profiles
        self_config['_locked'] = self._locked
        return self_config

    @classmethod
    def from_config(cls, config_dict, read_only=False):
        """
        Reconstruct a GeneratedModel2D from a configuration dictionary.

        Parameters
        ----------
        config_dict : dict
            Simulator configuration.

        Returns
        -------
        GeneratedModel2D
            Reconstructed GeneratedModel2D instance.
        """
        if not isinstance(config_dict, dict):
            raise TypeError("Expected a dictionary.")
        try:
            obj = cls.__new__(cls) 
            obj.gwt_depth = config_dict['properties_metadata']['gwt_depth']
            if config_dict['lit_domain'] is None:
                obj.lit_domain = None
            else:
                if read_only:
                    obj.lit_domain = LithologicalDomain2DReadOnly.from_config(config_dict['lit_domain'])
                else:                    
                    lm_type = config_dict['lit_domain']['lm_type']
                    if lm_type.startswith("from_interface_config"):
                        obj.lit_domain = LithologicalDomain2D.from_config(config_dict['lit_domain'])
                    else:
                        obj.lit_domain = LithologicalDomain2DFromObstruction2D.from_config(config_dict['lit_domain'])
                    
            obj.lit_id2material_dict = config_dict['properties_metadata']['lit_id2material_dict']
            obj.lit_order = config_dict['properties_metadata']['lit_order']
            obj.simulated_profiles = config_dict['simulated_profiles']
            obj._locked = config_dict['_locked']
            return obj

        except (KeyError, TypeError) as e:
            raise ValueError(f"Invalid config dictionary: {e}")
        
    def __eq__(self, other):
        """
        Compares this model with another for deep equality.

        Parameters
        ----------
        other : GeneratedModel2D
            Other model instance to compare.

        Returns
        -------
        bool
            True if models are equivalent in content, False otherwise.
        """
        if not isinstance(other, GeneratedModel2D):
            return NotImplemented
        
        return f.deep_object_equivalent(self, other, type_check=True)
      
        