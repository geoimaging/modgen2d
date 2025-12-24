"""
v2 includes miu and density profile generation
v3: a) distribution for Vp_mean and cov; Vp rather than miu for random_utility
    b) Added option to plot boundries in the figure_setting 
    c) boundaries settings now in spatial_sett, including "boundary_z_b...' from soil_sett
v4: includes depth pdf for utility
    includes options for interface generator
    Changed plot_boundary to its thickness
v5: includes random_init_options
v6: def profile_generator includes n_set, save_hdf5 options
    Plotting allowed for numpy profile_discrete
v7: v5 of random_utilities added. Incr_y, incr_z changed to del_y_refined, del_z_refined
v8: Corrected: Summarize concat, vp/vn min max
v8: Faster layer_id technique used in spatial_sim_v4

v9: code cleaning with a)pandas to numpy
"""
import h5py
import warnings
# from IPython.display import clear_output
import numpy as np
import geomodgen2d.general_functions as f
from geomodgen2d.spatial_simulator2d import SpatialSimulator2D
from geomodgen2d.generated_model2d import GeneratedModel2D
from geomodgen2d.lithological_domain2d_collection import LithologicalDomain2DCollection
from geomodgen2d.main_properties import MainPropertiesConfig

class GeneratedProfileCollection2DReadOnly:
    def __init__(self, main_properties_config_instance: MainPropertiesConfig, lithological_domain2d_collection: LithologicalDomain2DCollection, spatial_simulator2d_instance:SpatialSimulator2D):
        """
        Initializes the Generatedprofiles2D class.

        Args:
        - main_properties_config_instance (object): A MainPropertiesConfig instance.
        - spatial_simulator2d_instance (object): A SpatialSimulator2D's subclass instance.
        """
        if not main_properties_config_instance._locked:
            raise TypeError("main_properties_config_instance is not locked yet. Use .lock_and_generate_sample_properties first.")
        
        self._lit_id2material_dict = main_properties_config_instance.lit_id2material_dict
        self._sampled_properties = main_properties_config_instance.sampled_properties
        self._main_properties_unique_code = main_properties_config_instance.unique_code

        self._spatial_simulator2d_instance = spatial_simulator2d_instance
        
        self._generated_model2d_set = {}
        self._merged_generated_model2d = None
        self._generated_properties_list = []
        
        gwt = lithological_domain2d_collection.gwt_depth
        for set_name, lit_domain in lithological_domain2d_collection.lit_domain_set.items():
            self._generated_model2d_set[set_name] = GeneratedModel2D(lit_domain, gwt, self._lit_id2material_dict)

        self._merged_generated_model2d = GeneratedModel2D(lithological_domain2d_collection.merged_lit_domain, gwt, self._lit_id2material_dict)
        
    @property
    def generated_properties_list(self):
        return self._generated_properties_list
    
    @property
    def sampled_properties(self):
        return self._sampled_properties
    
    @property
    def lit_id2material_dict(self):
        return self._lit_id2material_dict
    
    @property
    def spatial_simulator2d_instance(self):
        return self._spatial_simulator2d_instance

    @property
    def main_properties_unique_code(self):
        return self._main_properties_unique_code
    
    def get_generated_model2d(self, set_name):
        return self._generated_model2d_set[set_name]
    
    @property
    def generated_model2d_set(self):
        return self._generated_model2d_set
    
    @property
    def merged_generated_model2d(self):
        return self._merged_generated_model2d
    
    @property
    def get_simulated_profiles(self, set_name):
        return self.generated_model2d_set[set_name].simulated_profiles

    @property
    def get_simulated_properties(self):
        keys = self.generated_model2d_set.keys()
        first_key = next(iter(keys), None)

        if first_key is None:
            return None

        return self.generated_model2d_set[first_key].simulated_profiles.keys()
    
    # def check(self):
    #     if self.material_domain_unique_code!=self.material_domain2d_instance.unique_code:
    #         raise ValueError("Material domain has been changed since definition of this GeneratedProfileCollection2D instance.")
        
    #     if not self.material_domain2d_instance._locked:
    #         raise ValueError("material domain2d instance has been unlocked. Redefine this class with locked one.")

    #TODO
    def save_to_hdf5(self, file_name, save_boundary_creator=False, save_lithological_domain=True):
        with h5py.File(file_name, 'w') as hf:
            # Save boundary_creator
            if save_boundary_creator:
                boundary_class = self.material_domain_instance.lithological_domain3D_class.boundary_class
                if boundary_class is None:
                    print("Warning. Boundary class is None in internal boundary class location. Generatedprofiles2D>MaterialDomain>LithologicalDomain3d>BoundaryCreator. So skipping.")
                else:
                    boundary_creator_group = hf.create_group("boundary_creator")
                    boundary_creator_group.create_dataset("x_ranges", data=boundary_class.x_ranges)
                    boundary_creator_group.create_dataset("y_ranges", data=boundary_class.y_ranges)
                    boundary_creator_group.create_dataset("z_ranges", data=boundary_class.z_ranges)
                    boundary_creator_group.create_dataset("n_layers", data=boundary_class.n_layers)
                    boundary_creator_group.create_dataset("b_array", data=boundary_class.boundary_array)
            
            # save_lithological_domain?
            if save_lithological_domain:
                lit_domain_class = self.material_domain_instance.lithological_domain3D_class
                lithological_domain_group = hf.create_group("lithological_domain")
                lithological_domain_group.create_dataset("x_ranges", data=lit_domain_class.x_ranges)
                lithological_domain_group.create_dataset("y_ranges", data=lit_domain_class.y_ranges)
                lithological_domain_group.create_dataset("z_ranges", data=lit_domain_class.z_ranges)
                lithological_domain_group.create_dataset("name", data=lit_domain_class.name)
                if lit_domain_class.gwt_depth is not None:
                    lithological_domain_group.create_dataset("gwt_depth", data=lit_domain_class.gwt_depth)
                s1 = convert_string_array_for_hdf5(self.lithological_domain3d_matrix)
                lithological_domain_group.create_dataset("layered_matrix", data=s1,
                                                        dtype=h5py.string_dtype(encoding='utf-8'))
                lithological_domain_group.create_dataset("lm_type", data=lit_domain_class.lm_type)
                lithological_domain_group.create_dataset("n_layers", data=lit_domain_class.n_layers)
                lithological_domain_group.create_dataset("overlap", data=lit_domain_class.overlap)
                lithological_domain_group.create_dataset("check", data=lit_domain_class.utils_description) 
                # lithological_domain_group.create_dataset("desc", data=lit_domain_class.utils_description)  # Dont know why, but this is not being saved. rather no addn being saved. so for now using check for utils_desc.
                lithological_domain_group.create_dataset("added_prefix", data=lit_domain_class.utils_description)
                print(lit_domain_class.utils_description)
                
            # save_materialDomain and save_generated_profiles
            self_group = hf.create_group("generated_profiles")
            self_group.create_dataset("x_ranges", data=self.x_ranges)
            self_group.create_dataset("y_ranges", data=self.y_ranges)
            self_group.create_dataset("z_ranges", data=self.z_ranges)
            lit_id2material_dict_group = self_group.create_group("lit_id2material_dict")
            for key, vals in self.lit_id2material_dict.items():
                lit_id2material_dict_group.create_dataset(key, data=convert_string_array_for_hdf5(vals))

            sampled_properties_group = self_group.create_group("sampled_properties")
            save_dict_to_hdf5(self.sampled_properties, sampled_properties_group)
            self_group.create_dataset('lithological_domain3d_matrix', data = convert_string_array_for_hdf5(self.lithological_domain3d_matrix), dtype=h5py.string_dtype(encoding='utf-8'))
            if self.gwt_depth is not None:
                self_group.create_dataset('gwt_depth', data=self.gwt_depth)

            all_generated_profiles_group = self_group.create_group("all_generated_profiles")
            save_dict_to_hdf5(self.all_generated_profiles, all_generated_profiles_group)
                
        print(f"Data saved to {file_name}")
        
class GeneratedProfileCollection2D(GeneratedProfileCollection2DReadOnly):
    def __init__(self, main_properties_config_instance: MainPropertiesConfig, lithological_domain2d_collection: LithologicalDomain2DCollection, spatial_simulator2d_instance:SpatialSimulator2D):
        """
        Initializes the Generatedprofiles2D class.

        Args:
        - main_properties_config_instance (object): A MainPropertiesConfig instance.
        - spatial_simulator2d_instance (object): A SpatialSimulator2D's subclass instance.
        """
        super().__init__(main_properties_config_instance, lithological_domain2d_collection, spatial_simulator2d_instance)
    
    def delete_simulated_property_profile(self, property_name):
        if property_name not in self.get_simulated_properties:
            raise ValueError(f"{property_name} is not generated yet. Generated Keys: {self.get_simulated_properties}")
        for _, set_name in enumerate(self.generated_model2d_set.keys()):
            self._generated_model2d_set[set_name].simulated_profiles.pop(property_name, None)
        self._merged_generated_model2d.simulated_profiles.pop(property_name, None)
    
    def simulate_zvals_property_profile(self, zvals_property_name,  
                                            generate_non_spatial_profile=False, 
                                            ignore_lithological_ids=['X'], simulated_val_for_ignored_lit_property=-99999):
        
        sampled_properties = self.sampled_properties
        if zvals_property_name in sampled_properties.keys():
                raise ValueError(f"zvals: {zvals_property_name} must NOT be in sampled_properties keys. Available {sampled_properties.keys()}. For z_vals generation. use generate_z_vals = True")
        
        if zvals_property_name in self.get_simulated_properties:
            raise ValueError(f"{zvals_property_name} already generated. Generated Keys: {self.get_simulated_properties}")
             
        for order_id, set_name in enumerate(self.generated_model2d_set.keys()):
            
            generated_model_instance = self.generated_model2d_set[set_name]
            if not isinstance(generated_model_instance, GeneratedModel2D):
                raise TypeError(f"All models in model set must be GeneratedModel2D instance. Issue in {set_name}")
            
            lit_domain = generated_model_instance.lit_domain
            
            #Check order_id and check locked
            
            gwt_depth = generated_model_instance.gwt_depth
            #TODO make sure??
            
            simulated_profile = self.spatial_simulator2d_instance.simulate_zvals_lit_profile_from_lithological_domain(
                lit_domain, gwt_depth, generate_non_spatial_profile, ignore_lithological_ids, simulated_val_for_ignored_lit_property
            )
            self._generated_model2d_set[set_name].simulated_profiles[zvals_property_name] = simulated_profile
            
            merged_domain = self.merged_generated_model2d.lit_domain
            simulated_profile_remeshed = f.remeshing_2D_matrix(
                x_old=lit_domain.domain.x_centers,
                x_new=merged_domain.domain.x_centers,
                z_old=lit_domain.domain.z_centers,
                z_new=merged_domain.domain.z_centers,
                matrix_2d=simulated_profile,
                interp_method='nearest'
            )
            if order_id==0:
                simulated_profile_merged = simulated_profile_remeshed
            else:
                # Masks to identify non-'none' values
                mask_other_non_none = ~np.isin(simulated_profile_remeshed, simulated_val_for_ignored_lit_property)
                simulated_profile_merged = np.where(mask_other_non_none, simulated_profile_remeshed, simulated_profile_merged)  # (replace it with that of other class even if merged already have values, i.e prioritize other class)
            
        # self.check_simulated_profile(simulated_profile_merged)
        self._merged_generated_model2d.simulated_profiles[zvals_property_name] = simulated_profile_merged
    
    def simulate_profile_from_zvals_property_profile(self, main_property_name, zvals_property_name, 
                                            ignore_lithological_ids=['X'], simulated_val_for_ignored_lit_property=-99999,
                                            min_val = None, max_val = None, warn_cliping = False):
        
        sampled_properties = self.sampled_properties
        if zvals_property_name not in self.get_simulated_properties:
            raise ValueError(f"{zvals_property_name} must be generated first. Generated Keys: {self.get_simulated_properties}")
        
        if main_property_name not in sampled_properties.keys():
                raise ValueError(f"{main_property_name} must be in sampled_properties keys. Available {sampled_properties.keys()}. For z_vals generation. use generate_z_vals = True")
            
        if main_property_name in self.get_simulated_properties:
            raise ValueError(f"{main_property_name} already generated. Generated Keys: {self.get_simulated_properties}")
             
        for order_id, set_name in enumerate(self.generated_model2d_set.keys()):
            
            generated_model_instance = self.generated_model2d_set[set_name]
            if not isinstance(generated_model_instance, GeneratedModel2D):
                raise TypeError(f"All models in model set must be GeneratedModel2D instance. Issue in {set_name}")
            
            lit_domain = generated_model_instance.lit_domain
            #Check order_id and check locked
            
            gwt_depth = generated_model_instance.gwt_depth
            #TODO make sure??
            
            simulated_zvals_lit_profile = self._generated_model2d_set[set_name].simulated_profiles[zvals_property_name]
            processed_property_dict = self.sampled_properties[main_property_name]
            
            simulated_profile = self.spatial_simulator2d_instance.simulate_profile_from_zvals_lit_profile(
                simulated_zvals_lit_profile, lit_domain,
                processed_property_dict, gwt_depth, warn_inconsistent_stdev = True,
                ignore_lithological_ids=ignore_lithological_ids,
                simulated_val_for_ignored_lit_property=simulated_val_for_ignored_lit_property)
            
            simulated_profile = self.clip_simulated_profile(simulated_profile, min_val, max_val, warn_cliping)      
            self._generated_model2d_set[set_name].simulated_profiles[main_property_name] = simulated_profile
            
            merged_domain = self.merged_generated_model2d.lit_domain
            simulated_profile_remeshed = f.remeshing_2D_matrix(
                x_old=lit_domain.domain.x_centers,
                x_new=merged_domain.domain.x_centers,
                z_old=lit_domain.domain.z_centers,
                z_new=merged_domain.domain.z_centers,
                matrix_2d=simulated_profile,
                interp_method='nearest'
            )
            
            if order_id==0:
                simulated_profile_merged = simulated_profile_remeshed
            else:
                # Masks to identify non-'none' values
                mask_other_non_none = ~np.isin(simulated_profile_remeshed, simulated_val_for_ignored_lit_property)
                simulated_profile_merged = np.where(mask_other_non_none, simulated_profile_remeshed, simulated_profile_merged)  # (replace it with that of other class even if merged already have values, i.e prioritize other class)
            
        # self.check_simulated_profile(simulated_profile_merged)
        simulated_profile_merged = self.clip_simulated_profile(simulated_profile_merged, 
                                                               min_val, max_val, raise_error=True)      
        
        self._merged_generated_model2d.simulated_profiles[main_property_name] = simulated_profile_merged

    @staticmethod
    def clip_simulated_profile(simulated_profile, min_val=None, max_val=None, warn=False, raise_error=False):
        if min_val is not None:
            assert isinstance(min_val, (int, float)), f"min_val must be a float/int. Provided {type(min_val)} : {min_val}"
        else: 
            min_val = np.min(simulated_profile)
            
        if max_val is not None:
            assert isinstance(max_val, (int, float)), f"max_val must be a float/int. Provided {type(max_val)} : {max_val}"
        else:
            max_val = np.max(simulated_profile)

        # Check if any values will be clipped
        values_outside_range = np.sum((simulated_profile < min_val) | (simulated_profile > max_val))

        if values_outside_range > 0:
            if warn:
                warnings.warn(f"Warning: {values_outside_range} values were clipped to the range [{min_val}, {max_val}]. Actual Range [{np.min(simulated_profile)}, {np.max(simulated_profile)}]")
            if raise_error:
                raise ValueError(f"{values_outside_range} values were clipped to the range [{min_val}, {max_val}]. Actual Range [{np.min(simulated_profile)}, {np.max(simulated_profile)}]")
                            
        simulated_profile = np.clip(simulated_profile, min_val, max_val)
        return simulated_profile
        
    def simulate_property_profile(
        self, main_property_name, 
        ignore_lithological_ids=['X'], simulated_val_for_ignored_lit_property=-99999,
        min_val = None, max_val = None, warn_cliping = False):
        """
        Generates a spatial profile by merging boundary and utility layer matrices.

        Args:
        - coarsed_coordinate_checked_class (object): Class to handle coarsed coordinate checking.
        - main_property_name (str): Property ID for the profile. Defaults to 'z_vals'.
        - generate_z_vals (bool, optional): Whether to generate z values. Defaults to True.
        - ignore_lithological_ids: list : Lithological ids to ignore during simulation. All values at these ids will have value -99999. 
        - min_val, max_val (float/int/None): Optional bounds to clip the property values. Values below min_val or above max_val will be truncated accordingly. These limits are not applied to z-values.
        """
        for trial in range(10000):
            zvals_property_name = f"___z_vals_{trial}"
            if zvals_property_name not in self.get_simulated_properties:
                break
       
        self.simulate_zvals_property_profile(
            zvals_property_name, generate_non_spatial_profile=False, 
            ignore_lithological_ids=ignore_lithological_ids, 
            simulated_val_for_ignored_lit_property=simulated_val_for_ignored_lit_property)
        
        self.simulate_profile_from_zvals_property_profile(
            main_property_name, zvals_property_name, 
            ignore_lithological_ids=ignore_lithological_ids, 
            simulated_val_for_ignored_lit_property=simulated_val_for_ignored_lit_property,
            min_val = min_val, max_val = max_val, warn_cliping = warn_cliping)
        
        self.delete_simulated_property_profile(zvals_property_name)
    
    #TODO
    def generate_profile_from_array(self, main_property_name, numpy_array, force_edit=False):
        """
        Generates a profile from a given numpy array.

        Args:
        - main_property_name (str): Property ID to assign to the generated profile.
        - numpy_array (ndarray): The array to generate the profile from.
        - force_edit (bool, optional): Whether to force edit if the profile exists. Defaults to False.
        """
        if not force_edit:
            if main_property_name in self.get_simulated_properties:
                raise ValueError(f"{main_property_name} already generated. Generated Keys: {self.get_simulated_properties}")
        simulated_profile = numpy_array
        # self.check_simulated_profile(simulated_profile)
        self._all_generated_profiles[main_property_name] = simulated_profile
    # save_generated_profiles
        
def save_dict_to_hdf5(d, parent_group):
    for key, value in d.items():
        if isinstance(value, dict):
            # Create a subgroup for nested dictionaries
            group = parent_group.create_group(key)
            save_dict_to_hdf5(value, group)
        else:
            # Save the value as a dataset
            parent_group.create_dataset(key, data=value)

def convert_string_array_for_hdf5(string_array):
    return np.array([s.encode('utf-8') for s in np.array(string_array).flatten()])#, np.array(string_array.shape)

