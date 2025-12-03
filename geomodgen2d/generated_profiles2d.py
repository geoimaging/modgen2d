# """
# v2 includes miu and density profile generation
# v3: a) distribution for Vp_mean and cov; Vp rather than miu for random_utility
#     b) Added option to plot boundries in the figure_setting 
#     c) boundaries settings now in spatial_sett, including "boundary_z_b...' from soil_sett
# v4: includes depth pdf for utility
#     includes options for interface generator
#     Changed plot_boundary to its thickness
# v5: includes random_init_options
# v6: def profile_generator includes n_set, save_hdf5 options
#     Plotting allowed for numpy profile_discrete
# v7: v5 of random_utilities added. Incr_y, incr_z changed to del_y_refined, del_z_refined
# v8: Corrected: Summarize concat, vp/vn min max
# v8: Faster layer_id technique used in spatial_sim_v4

# v9: code cleaning with a)pandas to numpy
# """
# import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm
# import matplotlib.cm as cm
# import h5py
# import time
# # from IPython.display import clear_output
# import numpy as np
# from geomodgen2d.material_domain2d import MaterialDomain
# import geomodgen2d.general_functions as f
# import geomodgen2d.spatial_simulation2d as spatial_simulation2d
    
# class GeneratedProfiles2DFunctions:
#     """
#     A class for generating and managing geotechnical profiles, including non-spatial, spatial, and profiles derived from arrays.
#     """
#     def __init__(self, material_domain_instance: MaterialDomain, theta_x:float, theta_z:float, rnd_no=np.random.default_rng()):
#         """
#         Initializes the Generatedprofiles2D class.

#         Args:
#         - material_domain_instance (object): A MaterialDomain instance.
#         - theta_x, theta_z (float): Parameter for spatial generation in the x-direction, z-direction.
#         - rnd_no (int): Random seed number for stochastic generation.
#         """
#         self.material_domain_instance = material_domain_instance
        
#         if material_domain_instance is not None:
#             assert isinstance(material_domain_instance, MaterialDomain), "material_domain_instance must be a MaterialDomain instance"
#             self.lithological_domain2D_instance_list = material_domain_instance.lithological_domain2D_instance_list
#             self.lithological_domain2D_combined_w_min_spacing = material_domain_instance.lithological_domain2D_combined_w_min_spacing
#             self.span_x, self.span_z = material_domain_instance.span_x, material_domain_instance.span_z
#             self.min_del_x, self.min_del_z = material_domain_instance.min_del_x, material_domain_instance.min_del_z
#             # self.x_ranges, self.y_ranges, self.z_ranges = material_domain_instance.x_ranges, material_domain_instance.y_ranges, material_domain_instance.z_ranges
#             self.lithological_domain2D_combined_w_min_spacing = material_domain_instance.lithological_domain2D_combined_w_min_spacing    
#             self.lit_id2material_dict = material_domain_instance.lit_id2material_dict
#             self.sampled_properties = material_domain_instance.sampled_properties
#             self.gwt_depth = material_domain_instance.gwt_depth
            
#         self.read_only = False
#         self.theta_x = theta_x
#         self.theta_z = theta_z
#         self.rnd_no = rnd_no
#         self._all_generated_profiles = {}

#     def check_simulated_profile(self, simulated_profile):
#         """
#         Checks that the simulated profile has the correct shape and contains no NaN values.

#         Args:
#         - simulated_profile (ndarray): The simulated profile to check.
#         """
#         assert (simulated_profile.shape == self.lithological_domain2D_combined_w_min_spacing.layered_matrix.shape), f"ERROR: shape of simulated profile and layered matrix does not match. {simulated_profile.shape} != {self.lithological_domain2D_combined_w_min_spacing.layered_matrix.shape}"
#         contains_nan = np.isnan(simulated_profile).any()

#         assert contains_nan==0, "the simulated profile cannot have nan-values"
        
#         # Check if the profile contains any values of -99999
#         contains_negative_value = (simulated_profile == -99999).any()
#         if contains_negative_value:
#             print("Warning: Few numbers in simulated profile is -99999, denoting ignored layers.")
                 
#     def plot2d(self, main_property_name, ax=None, scatter_point_size=0, 
#                warning=True, 
#                vlog = False, vmin=None, vmax=None, cmap='gist_earth_r', 
#                title = 'auto', legend = True, legend_label = None):
#         """
#         Plots a 2D section of the layered matrix.

#         Parameters:
#             ax: The matplotlib axes object for the plot (default is None, which creates a new figure).
#             scatter_point_size: Scatter points size (Shows points before interpolation.)
#             warning: Print warning message for odd section plots.
#             scatter_point_size: Scatter points size (Shows points before interpolation.)
#             vlog (boolean, optional): Use log normalization
#             vmin (float, optional): Minimum value for colormap scaling. Defaults to None.
#             vmax (float, optional): Maximum value for colormap scaling. Defaults to None.
#             color_map: A dictionary that defines the color map for the values in the matrix.
#         """
#         if ax is None:
#             fig,ax = plt.subplots()

#         gen_data = self._all_generated_profiles[main_property_name]
        
#         if self.lithological_domain2D_combined_w_min_spacing.dim == 1:
#             x_ranges_plt = [-self.span_z/10, self.span_z/10]
#         else:
#             x_ranges_plt = [0, self.span_x]
        
#         extent=[x_ranges_plt[0], x_ranges_plt[1], self.span_z, 0]
        
#         data = gen_data
        
        
#         # Create a colormap from the color mapping
#         if vmin is None:
#             vmin = np.min(data)

#         if vmax is None:
#             vmax = np.max(data)
            
#         if vlog:
#             norm = LogNorm(vmin=vmin, vmax=vmax)
#             cax = ax.imshow(data, norm=norm, cmap=cmap, extent=extent, interpolation='none') 
#         else:
#             cax = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, interpolation='none') 
        
#         # if scatter_point_size!=0:
#         #     ax.scatter(x_data.flatten(), y_data.flatten(), c = data.flatten(), edgecolors='k', s=scatter_point_size) # Not correct perfectly yet.
       
#         # Colorbar
#         if legend:
#             cbar = plt.colorbar(cax, ax=ax, shrink=0.6, aspect=20, pad=0.1)
#             cbar.set_label(legend_label)
        
#         return ax, vmin, vmax
    
#     @property
#     def all_generated_profiles(self):
#         return self._all_generated_profiles
    
# class GeneratedProfiles2D(GeneratedProfiles2DFunctions):
#     def __init__(self, material_domain_instance: MaterialDomain, theta_x:float, theta_z:float, rnd_no=np.random.default_rng()):
#         """
#         Initializes the LithogolicalDomain3D instance with given spatial limits, and spacing.
        
#         Parameters:
#         span_x, span_y, span_z : float
#             The upper limit for the x, y, and z-coordinate range.
#         del_x, del_y, del_z : float
#             The spacing interval for x, y, and z-coordinates.
#         name: str
#             The name of lithologicaldomain
#         """
#         assert material_domain_instance is not None, "material_domain_instance cannot be None for Generatedprofiles2D class."
#         super().__init__(material_domain_instance, theta_x, theta_z, rnd_no)
        
#     def generate_spatial(self, main_property_name, generate_z_vals = False, ignore_lithological_ids=['X'], min_val=None, max_val=None):
#         """
#         Generates a spatial profile by merging boundary and utility layer matrices.

#         Args:
#         - coarsed_coordinate_checked_class (object): Class to handle coarsed coordinate checking.
#         - main_property_name (str, optional): Property ID for the profile. Defaults to 'z_vals'.
#         - generate_z_vals (bool, optional): Whether to generate z values. Defaults to True.
#         - ignore_lithological_ids: list : Lithological ids to ignore during simulation. All values at these ids will have value -99999. 
#         - min_val, max_val (float/int/None): Optional bounds to clip the property values. Values below min_val or above max_val will be truncated accordingly. These limits are not applied to z-values.
#         """
#         # Assert layer_matrix_class_from_boundary and layer_matrix_class_from_utils combines to create self.layer_matrix_class
#         if generate_z_vals:
#             processed_property_dict = None
#             assert main_property_name not in self.sampled_properties.keys(), f"{main_property_name} must not be in sampled_properties keys for z-vals generation. Available {self.sampled_properties.keys()}"
#         else:
#             assert main_property_name in self.sampled_properties.keys(), f"{main_property_name} must be in sampled_properties keys. Available {self.sampled_properties.keys()}. For z_vals generation. use generate_z_vals = True"
#             processed_property_dict = self.sampled_properties[main_property_name]
            
#         if main_property_name in self._all_generated_profiles.keys():
#             raise ValueError(f"{main_property_name} already generated. Generated Keys: {self._all_generated_profiles.keys()}")
            
#         i=0
#         for lithological_domain2D_class in self.lithological_domain2D_instance_list:
#             simulated_profile = spatial_simulation2d.spatial_profile_gen(self.theta_x, self.theta_z, lithological_domain2D_class, self.gwt_depth, self.rnd_no, processed_property_dict=processed_property_dict, ignore_lithological_ids=ignore_lithological_ids)
#             print(simulated_profile.shape,lithological_domain2D_class.layered_matrix.shape)
#             simulated_profile = f.upsample_2d_blocks(lithological_domain2D_class.x_ranges, lithological_domain2D_class.z_ranges, 
#                                                      self.lithological_domain2D_combined_w_min_spacing.x_ranges, self.lithological_domain2D_combined_w_min_spacing.z_ranges, simulated_profile)
#             print(simulated_profile.shape)
            
#             if i==0:
#                 simulated_profile_merged = simulated_profile
#             else:
#                 # Masks to identify non-'none' values
#                 mask_other_non_none = ~np.isin(simulated_profile, [-99999])
#                 simulated_profile_merged = np.where(mask_other_non_none, simulated_profile, simulated_profile_merged)  # (replace it with that of other class even if merged already have values, i.e prioritize other class)
#             i+=1
            
#         self.check_simulated_profile(simulated_profile_merged)

#         if min_val is not None:
#             assert isinstance(min_val, (int, float)), f"min_val must be a float/int. Provided {type(min_val)} : {min_val}"
#         else: 
#             min_val = np.min(simulated_profile_merged)
#         if max_val is not None:
#             assert isinstance(max_val, (int, float)), f"max_val must be a float/int. Provided {type(max_val)} : {max_val}"
#         else:
#             max_val = np.max(simulated_profile_merged)

#         # Check if any values will be clipped
#         values_outside_range = np.sum((simulated_profile_merged < min_val) | (simulated_profile_merged > max_val))
        
#         if values_outside_range > 0:
#             print(f"Warning: {values_outside_range} values were clipped to the range [{min_val}, {max_val}]. Actual Range [{np.min(simulated_profile_merged)}, {np.max(simulated_profile_merged)}]")
            
#         simulated_profile_merged = np.clip(simulated_profile_merged, min_val, max_val)
#         self._all_generated_profiles[main_property_name] = simulated_profile_merged
        
#     def generate_spatial_from_z_vals(self, main_property_name, z_vals_property_name, ignore_lithological_ids=[], min_val=None, max_val=None):
#         """
#         Generates a spatial profile from pre-existing z-values (z_vals_property_name). So, z_vals_property_name must already be generated. 
#         Note: Here, new generated profile = mean + stdev * z_vals

#         Args:
#         - main_property_name (str): Property ID to generate.
#         - z_vals_property_name (str): Generated z-vals
#         - ignore_lithological_ids: list : Lithological ids to ignore during simulation. All values at these ids will have value -99999. 
#         - min_val, max_val (float/int/None): Optional bounds to clip the property values. Values below min_val or above max_val will be truncated accordingly. These limits are not applied to z-values.
#         """
#         if z_vals_property_name not in self._all_generated_profiles.keys():
#             raise ValueError(f"{z_vals_property_name} not generated yet. Generate it first. Generated Keys: {self._all_generated_profiles.keys()}")
        
#         assert main_property_name in self.sampled_properties.keys(), f"{main_property_name} must be in sampled_properties keys. Available {self.sampled_properties.keys}"
            
#         if main_property_name in self._all_generated_profiles.keys():
#             raise ValueError(f"{main_property_name} already generated. Generated Keys: {self._all_generated_profiles.keys()}")

#         mean_profile = spatial_simulation2d.non_spatial_simulation(self.lithological_domain2D_combined_w_min_spacing, self.gwt_depth, self.sampled_properties[main_property_name], ignore_lithological_ids, prof_type='mean', warn_non_zero_stdev=False)
#         std_profile = spatial_simulation2d.non_spatial_simulation(self.lithological_domain2D_combined_w_min_spacing, self.gwt_depth, self.sampled_properties[main_property_name], ignore_lithological_ids, prof_type='stdev/cov', warn_non_zero_stdev=False)
        
#         simulated_profile = mean_profile + self._all_generated_profiles[z_vals_property_name]*std_profile
#         self.check_simulated_profile(simulated_profile)

#         if min_val is not None:
#             assert isinstance(min_val, (int, float)), f"min_val must be a float/int. Provided {type(min_val)} : {min_val}"
#         else: 
#             min_val = np.min(simulated_profile)
#         if max_val is not None:
#             assert isinstance(max_val, (int, float)), f"max_val must be a float/int. Provided {type(max_val)} : {max_val}"
#         else:
#             max_val = np.max(simulated_profile)

#         # Check if any values will be clipped
#         values_outside_range = np.sum((simulated_profile < min_val) | (simulated_profile > max_val))
        
#         if values_outside_range > 0:
#             print(f"Warning: {values_outside_range} values were clipped to the range [{min_val}, {max_val}]. Actual Range [{np.min(simulated_profile_merged)}, {np.max(simulated_profile_merged)}]")

#         simulated_profile_merged = np.clip(simulated_profile, min_val, max_val)
#         self._all_generated_profiles[main_property_name] = simulated_profile

#     def generate_profile_from_array(self, main_property_name, numpy_array, force_edit=False):
#         """
#         Generates a profile from a given numpy array.

#         Args:
#         - main_property_name (str): Property ID to assign to the generated profile.
#         - numpy_array (ndarray): The array to generate the profile from.
#         - force_edit (bool, optional): Whether to force edit if the profile exists. Defaults to False.
#         """
#         if not force_edit:
#             if main_property_name in self._all_generated_profiles.keys():
#                 raise ValueError(f"{main_property_name} already generated. Generated Keys: {self._all_generated_profiles.keys()}")
#         simulated_profile = numpy_array
#         self.check_simulated_profile(simulated_profile)
#         self._all_generated_profiles[main_property_name] = simulated_profile
#     # save_generated_profiles
        
#     def save_to_hdf5(self, file_name, save_boundary_creator=False, save_lithological_domain=True):
#         with h5py.File(file_name, 'w') as hf:
#             # Save boundary_creator
#             if save_boundary_creator:
#                 boundary_class = self.material_domain_instance.lithological_domain3D_class.boundary_class
#                 if boundary_class is None:
#                     print("Warning. Boundary class is None in internal boundary class location. Generatedprofiles2D>MaterialDomain>LithologicalDomain3d>BoundaryCreator. So skipping.")
#                 else:
#                     boundary_creator_group = hf.create_group("boundary_creator")
#                     boundary_creator_group.create_dataset("x_ranges", data=boundary_class.x_ranges)
#                     boundary_creator_group.create_dataset("y_ranges", data=boundary_class.y_ranges)
#                     boundary_creator_group.create_dataset("z_ranges", data=boundary_class.z_ranges)
#                     boundary_creator_group.create_dataset("n_layers", data=boundary_class.n_layers)
#                     boundary_creator_group.create_dataset("b_array", data=boundary_class.boundary_array)
            
#             # save_lithological_domain?
#             if save_lithological_domain:
#                 lit_domain_class = self.material_domain_instance.lithological_domain3D_class
#                 lithological_domain_group = hf.create_group("lithological_domain")
#                 lithological_domain_group.create_dataset("x_ranges", data=lit_domain_class.x_ranges)
#                 lithological_domain_group.create_dataset("y_ranges", data=lit_domain_class.y_ranges)
#                 lithological_domain_group.create_dataset("z_ranges", data=lit_domain_class.z_ranges)
#                 lithological_domain_group.create_dataset("name", data=lit_domain_class.name)
#                 if lit_domain_class.gwt_depth is not None:
#                     lithological_domain_group.create_dataset("gwt_depth", data=lit_domain_class.gwt_depth)
#                 s1 = convert_string_array_for_hdf5(self.lithological_domain3d_matrix)
#                 lithological_domain_group.create_dataset("layered_matrix", data=s1,
#                                                         dtype=h5py.string_dtype(encoding='utf-8'))
#                 lithological_domain_group.create_dataset("lm_type", data=lit_domain_class.lm_type)
#                 lithological_domain_group.create_dataset("n_layers", data=lit_domain_class.n_layers)
#                 lithological_domain_group.create_dataset("overlap", data=lit_domain_class.overlap)
#                 lithological_domain_group.create_dataset("check", data=lit_domain_class.utils_description) 
#                 # lithological_domain_group.create_dataset("desc", data=lit_domain_class.utils_description)  # Dont know why, but this is not being saved. rather no addn being saved. so for now using check for utils_desc.
#                 lithological_domain_group.create_dataset("added_prefix", data=lit_domain_class.utils_description)
#                 print(lit_domain_class.utils_description)
                
#             # save_materialDomain and save_generated_profiles
#             self_group = hf.create_group("generated_profiles")
#             self_group.create_dataset("x_ranges", data=self.x_ranges)
#             self_group.create_dataset("y_ranges", data=self.y_ranges)
#             self_group.create_dataset("z_ranges", data=self.z_ranges)
#             lit_id2material_dict_group = self_group.create_group("lit_id2material_dict")
#             for key, vals in self.lit_id2material_dict.items():
#                 lit_id2material_dict_group.create_dataset(key, data=convert_string_array_for_hdf5(vals))

#             sampled_properties_group = self_group.create_group("sampled_properties")
#             save_dict_to_hdf5(self.sampled_properties, sampled_properties_group)
#             self_group.create_dataset('lithological_domain3d_matrix', data = convert_string_array_for_hdf5(self.lithological_domain3d_matrix), dtype=h5py.string_dtype(encoding='utf-8'))
#             if self.gwt_depth is not None:
#                 self_group.create_dataset('gwt_depth', data=self.gwt_depth)

#             all_generated_profiles_group = self_group.create_group("all_generated_profiles")
#             save_dict_to_hdf5(self.all_generated_profiles, all_generated_profiles_group)
                
#         print(f"Data saved to {file_name}")
        
# class Generatedprofiles2DReadOnly(Generatedprofiles2DFunctions):
#     def __init__(self, generated_profiles_dict:dict):
#         """
#         Initializes the LithogolicalDomain3D instance with given spatial limits, and spacing.
        
#         Parameters:
#         generated_profiles_dict: dict
#             The dictionary from loaded file (read_only)
#         """
#         self.read_only = True
#         super().__init__(None, 0, 0, 0, None)
        
#         span_x, span_y, span_z, del_x, del_y, del_z = f.coordinate_vars(generated_profiles_dict['x_ranges'], generated_profiles_dict['y_ranges'], generated_profiles_dict['z_ranges'])
#         self.lithological_domain3D_class = None
#         self.span_x, self.span_y, self.span_z = span_x, span_y, span_z
#         self.del_x, self.del_y, self.del_z = del_x, del_y, del_z
#         self.x_ranges, self.y_ranges, self.z_ranges = generated_profiles_dict['x_ranges'], generated_profiles_dict['y_ranges'], generated_profiles_dict['z_ranges']
#         self.lithological_domain3d_matrix = generated_profiles_dict['lithological_domain3d_matrix']
#         self.lithological_domain3d_matrix = np.array([s.decode('utf-8') for s in self.lithological_domain3d_matrix]).reshape((len(self.z_ranges), len(self.y_ranges), len(self.x_ranges)))
        
#         self.lit_id2material_dict = generated_profiles_dict['lit_id2material_dict']
#         self.sampled_properties = generated_profiles_dict['sampled_properties']
#         if 'gwt_depth' in generated_profiles_dict.keys():
#             gwt_depth = generated_profiles_dict['gwt_depth']
#         else:
#             gwt_depth = None
            
#         self.gwt_depth = gwt_depth
            
#         self.read_only = False
#         self._all_generated_profiles = generated_profiles_dict['all_generated_profiles']
        
# def save_dict_to_hdf5(d, parent_group):
#     for key, value in d.items():
#         if isinstance(value, dict):
#             # Create a subgroup for nested dictionaries
#             group = parent_group.create_group(key)
#             save_dict_to_hdf5(value, group)
#         else:
#             # Save the value as a dataset
#             parent_group.create_dataset(key, data=value)

# def convert_string_array_for_hdf5(string_array):
#     return np.array([s.encode('utf-8') for s in np.array(string_array).flatten()])#, np.array(string_array.shape)

