import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
# from IPython.display import clear_output
import numpy as np
import geomodgen2d.general_functions as f
from geomodgen2d.lithological_domain2d import LithologicalDomain2D
import warnings

class GeneratedModel2D:
    def __init__(self, lithological_domain_instance:LithologicalDomain2D, gwt_depth, lit_id2material_dict):
        self.lit_domain = lithological_domain_instance
        self.lit_order = lithological_domain_instance.lit_order
        self.gwt_depth = gwt_depth
        self.lit_id2material_dict = {}
        self.simulated_profiles = {}
        self._locked = False
        
        lithological_domain_instance.check_shape()
        
        unique_ids = np.unique(self.lit_domain.lithological_matrix.astype(str))
        unique_ids = unique_ids[unique_ids!="X"]
        
        # Check that all unique IDs are keys in lit_id2material
        missing_keys = [uid for uid in unique_ids if uid not in lit_id2material_dict]
        if missing_keys:
            raise ValueError(f"The following lithological IDs are missing in lit_id2material_dict: {missing_keys}")
        
        # Filter out any extra keys in lit_id2material_dict
        filtered_dict = {k: v for k, v in lit_id2material_dict.items() if k in unique_ids}
        self.lit_id2material_dict = filtered_dict
    
    def check(self):
        """
        Checks that the simulated profiles has the correct shape and contains no NaN values.

        Args:
        - simulated_profile (ndarray): The simulated profile to check.
        """
        domain_shape = self.lit_domain.domain.shape
        lit_shape = self.lit_domain.lithological_matrix.shape
        if domain_shape != lit_shape:
            raise ValueError(f"Matrix shape mismatch. Domain shape {domain_shape} != lit_shape {lit_shape}.")
        
        for key,val in self.simulated_profiles.items():
            if domain_shape != val.shape:
                raise ValueError(f"Matrix shape mismatch. Domain shape {domain_shape} != lit_shape {val.shape} for property: {key}.")
    
        unique_ids = np.unique(self.lit_domain.lithological_matrix.astype(str))
        # Check that all unique IDs are keys in lit_id2material
        missing_keys = [uid for uid in unique_ids if uid not in self.lit_id2material_dict]
        if missing_keys:
            raise ValueError(f"The following lithological IDs are missing in lit_id2material_dict: {missing_keys}")

        for key,val in self.simulated_profiles.items():
            contains_nan = np.isnan(val).any()
            if contains_nan!=0:
                raise ValueError(f"The simulated profile for property {key} cannot have nan-values")
        
            # Check if the profile contains any values of -99999
            contains_negative_value = (val == -99999).any()
            if contains_negative_value:
                print("Warning: Few numbers in simulated profile for property {key} is -99999, denoting ignored layers.")
    
    def remesh(self, new_dx, new_dz):
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
        
        if self.lit_id2material_dict and use_lit_id2material_dict:
            lit_id2material_dict = self.lit_id2material_dict
        else:
            lit_id2material_dict = None 
        
        self.lit_domain.plot(ax=ax, discrete_point_size=discrete_point_size, legend=legend,
               id2material_dict = lit_id2material_dict, title=title, plot_interfaces=plot_interfaces,
               color_map = color_map)

    def plot_profile(self, main_property_name, ax=None, discrete_point_size=0, plot_gwt = True,
               vlog = False, vmin=None, vmax=None, cmap='gist_earth_r', 
               title = 'auto', legend = True, legend_label = None, legendkwargs_dict={}):
        """
        Plots a 2D section of the layered matrix.

        Parameters:
            ax: The matplotlib axes object for the plot (default is None, which creates a new figure).
            scatter_point_size: Scatter points size (Shows points before interpolation.)
            warning: Print warning message for odd section plots.
            scatter_point_size: Scatter points size (Shows points before interpolation.)
            vlog (boolean, optional): Use log normalization
            vmin (float, optional): Minimum value for colormap scaling. Defaults to None.
            vmax (float, optional): Maximum value for colormap scaling. Defaults to None.
            color_map: A dictionary that defines the color map for the values in the matrix.
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
        