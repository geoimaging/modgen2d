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
from geomodgen2d.spatial_simulator2d import CovarianceDecompositionSimulator, SpatialSimulator2D    
from geomodgen2d.generated_model2d import GeneratedModel2D
from geomodgen2d.lithological_domain2d import LithologicalDomain2DCollection, LithologicalDomain2DReadOnly
from geomodgen2d.main_properties import MainPropertiesConfig
from geomodgen2d.global_soil_interface_config import GlobalSoilInterfaceConfig
from geomodgen2d.metadata import __version__

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
        self._read_only = False # Initially read_only flag is False.
        
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

    def change_spatial_simulator_type(self, new_simulator_class):
        self._spatial_simulator2d_instance.change_spatial_simulator_type(new_simulator_class)
        
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
    
    # Check if all the simulated values ignoer is same in spatial simulator and here. also gwt??
    
    # To do: get merged and check if asme with the one generateed. If jot woo woo!!!

    @property
    def get_config(self):
        self_config = {}
        self_config['_read_only'] = self._read_only
        self_config['properties_metadata'] = {}
        self_config['properties_metadata']['_lit_id2material_dict'] = self._lit_id2material_dict
        self_config['properties_metadata']['_sampled_properties'] = self._sampled_properties
        self_config['properties_metadata']['_main_properties_unique_code'] = self._main_properties_unique_code
        self_config['properties_metadata']['_generated_properties_list'] = self._generated_properties_list
        self_config['_spatial_simulator2d_instance'] = self._spatial_simulator2d_instance.get_config
        
        self_config['_generated_model2d_set'] = {}
        for key, val in self._generated_model2d_set.items():
            self_config['_generated_model2d_set'][key] = val.get_config
        
        self_config['_merged_generated_model2d']=None
        if self._merged_generated_model2d is not None:
            self_config['_merged_generated_model2d']=self._merged_generated_model2d.get_config
        
        return self_config
    
    @classmethod
    def from_config(cls, config_dict, read_only=False, check_merged=False):
        if not isinstance(config_dict, dict):
            raise TypeError("Expected a dictionary.")
        # try:
        
        if config_dict['_read_only'] is True and read_only is False:
            raise ValueError("Given config_dict is intended for read_only = True, but attempted to read in read_only = False mode.")
        
        val = config_dict.get('_generated_model2d_set')
        if val is None and read_only is True:
            raise ValueError("Provided generated_model2d as None. Non-read_only (i.e., read_only = False) cannot be used for such cases.")
        
        obj = cls.__new__(cls) 
        obj._read_only = read_only
        obj._generated_properties_list = config_dict['properties_metadata']['_generated_properties_list']
        obj._lit_id2material_dict = config_dict['properties_metadata']['_lit_id2material_dict']
        obj._main_properties_unique_code = config_dict['properties_metadata']['_main_properties_unique_code']
        obj._sampled_properties = config_dict['properties_metadata']['_sampled_properties']
        obj._spatial_simulator2d_instance = CovarianceDecompositionSimulator.from_config(config_dict['_spatial_simulator2d_instance'])

        expected = CovarianceDecompositionSimulator.__name__
        actual = config_dict['_spatial_simulator2d_instance']['simulator_type_name']
        if expected != actual:
            warnings.warn(f"Loaded GeneratedProfileCollection2D have mismatched spatial simulator. Use .change_spatial_simulator_type({expected})",
            RuntimeWarning
        )
        
        obj._generated_model2d_set = {}  
        if val is not None:
            for key, val in val.items():
                obj._generated_model2d_set[key] = GeneratedModel2D.from_config(val)
        
        if config_dict['_merged_generated_model2d'] is None:
            obj._merged_generated_model2d = config_dict['_merged_generated_model2d']
        else:
            obj._merged_generated_model2d = GeneratedModel2D.from_config(config_dict['_merged_generated_model2d'])
        return obj

    def save_to_hdf5(self, file_name, hdf5_compression_level=0):
        to_save_dict = {
            'geomodgen2d_version': __version__,
            'save_read_only': False,
            'global_interface_config': GlobalSoilInterfaceConfig.get_config,
            'gen_model_2d_collection': self.get_config
        }
        
        with h5py.File(file_name, 'w') as hf:
            save_dict_to_hdf5(to_save_dict, hf, compression_level=hdf5_compression_level)
            
        print(f"Data saved to {file_name}")
        
    def save_to_hdf5_read_only(self, file_name, save_merged_only=True, save_interface=False, hdf5_compression_level=0):
        state_dict = self.get_config.copy()
        if save_merged_only:
            state_dict.pop('_generated_model2d_set', None)
                
        to_save_dict = {
            'geomodgen2d_version': __version__,
            'save_read_only': True,
            'global_interface_config': GlobalSoilInterfaceConfig.get_config,
            'gen_model_2d_collection': state_dict
        }
        
        if not save_interface:
            to_save_dict.pop('global_interface_config')
            
        with h5py.File(file_name, 'w') as hf:
            save_dict_to_hdf5(to_save_dict, hf, compression_level=hdf5_compression_level)
        
        print(f"Data saved to {file_name}")
        
    def save_to_hdf5_numpy(self, file_name, save_merged_only=True, save_interface=False, save_lithological_domain=False, save_properties_metadata=False, hdf5_compression_level=0):
        with h5py.File(file_name, 'w') as hf:
            to_save_dict = {
                'geomodgen2d_version': __version__,
            }
            save_dict_to_hdf5(to_save_dict, hf, compression_level=hdf5_compression_level)
            
            if save_interface:
                state_dict = GlobalSoilInterfaceConfig.get_config
                to_save_dict = {
                    'interfaces_matrix': state_dict['_discretized_interface2d_instance']['interfaces_matrix']
                }
                save_dict_to_hdf5(to_save_dict, hf, compression_level=hdf5_compression_level)
            
            gmodel2d_coll_group = hf.create_group("gen_model_2d_collection")
            state_dict = self.get_config
            if save_properties_metadata:
                properties_metadata = {
                    'properties_metadata':{'_lit_id2material_dict': state_dict['properties_metadata']['_lit_id2material_dict'],
                                           '_sampled_properties': state_dict['properties_metadata']['_sampled_properties']
                                           }
                }
                save_dict_to_hdf5(properties_metadata, gmodel2d_coll_group, compression_level=hdf5_compression_level)
                
            dict_group = gmodel2d_coll_group.create_group("_merged_generated_model2d")
            
            merged_generated_model2d_dict = {
                'properties_metadata': state_dict['_merged_generated_model2d']['properties_metadata'],
                'lit_domain': state_dict['_merged_generated_model2d']['lit_domain'],
                'simulated_profiles': state_dict['_merged_generated_model2d']['simulated_profiles'],
            }
            
            if not save_properties_metadata:
                merged_generated_model2d_dict.pop('properties_metadata')
            
            if save_lithological_domain:
                merged_generated_model2d_dict.pop('lit_domain')

            save_dict_to_hdf5(merged_generated_model2d_dict, dict_group, compression_level=hdf5_compression_level)
                
            if not save_merged_only:
                dict3_group = gmodel2d_coll_group.create_group("_generated_model2d_set")
                state_dict = state_dict['_generated_model2d_set']
                
                for keys, values in state_dict.items():
                    dict_group = dict3_group.create_group(keys)
                    
                    merged_generated_model2d_dict = {
                        'properties_metadata': state_dict[keys]['properties_metadata'],
                        'lit_domain': state_dict[keys]['lit_domain'],
                        'simulated_profiles': state_dict[keys]['simulated_profiles'],
                    }
                    
                    if not save_properties_metadata:
                        merged_generated_model2d_dict.pop('properties_metadata')
                    
                    if save_lithological_domain:
                        merged_generated_model2d_dict.pop('lit_domain')
                    
                    save_dict_to_hdf5(merged_generated_model2d_dict, dict_group, compression_level=hdf5_compression_level)
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
    
    @staticmethod
    def get_merged_simulated_profile(current_merged_lit_domain:LithologicalDomain2DReadOnly, current_merged_simulated_profile:np.ndarray,
                                     to_merge_lit_domain:LithologicalDomain2DReadOnly, to_merge_simulated_profile:np.ndarray, simulated_val_for_ignored_lit_property):
        to_merge_simulated_profile_remeshed = f.remeshing_2D_matrix(
            x_old=to_merge_lit_domain.domain.x_centers,
            x_new=current_merged_lit_domain.domain.x_centers,
            z_old=to_merge_lit_domain.domain.z_centers,
            z_new=current_merged_lit_domain.domain.z_centers,
            matrix_2d=to_merge_simulated_profile,
            interp_method='nearest'
        )
        
        if current_merged_simulated_profile is None:
            current_merged_simulated_profile = to_merge_simulated_profile_remeshed
        else:
            # Masks to identify non-'none' values
            mask_other_non_none = ~np.isin(to_merge_simulated_profile_remeshed, simulated_val_for_ignored_lit_property)
            simulated_profile_merged = np.where(mask_other_non_none, to_merge_simulated_profile_remeshed, current_merged_simulated_profile)  # (replace it with that of other class even if merged already have values, i.e prioritize other class)
        
        return simulated_profile_merged
   
    def simulate_zvals_property_profile(self, zvals_property_name,  
                                            generate_non_spatial_profile=False, 
                                            ignore_lithological_ids=['X']):
        
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
            
            if (generated_model_instance.simulated_val_for_ignored_lit_property != self.spatial_simulator2d_instance.simulated_val_for_ignored_lit_property):
                raise ValueError(
                    f"Inconsistent simulated_val_for_ignored_lit_property: "
                    f"{generated_model_instance.simulated_val_for_ignored_lit_property} "
                    f"!= {self.spatial_simulator2d_instance.simulated_val_for_ignored_lit_property}"
                )
            
            simulated_profile = self.spatial_simulator2d_instance.simulate_zvals_lit_profile_from_lithological_domain(
                lit_domain, gwt_depth, generate_non_spatial_profile, ignore_lithological_ids
            )
            self._generated_model2d_set[set_name].simulated_profiles[zvals_property_name] = simulated_profile
            
            merged_lit_domain = self.merged_generated_model2d.lit_domain
            if order_id==0:
                simulated_profile_merged = None
                
            simulated_profile_merged = self.get_merged_simulated_profile(
                order_id, current_merged_lit_domain=merged_lit_domain, current_merged_simulated_profile=simulated_profile_merged,
                to_merge_lit_domain = lit_domain, to_merge_simulated_profile=simulated_profile, 
                simulated_val_for_ignored_lit_property=generated_model_instance.simulated_val_for_ignored_lit_property
                )
            
        # self.check_simulated_profile(simulated_profile_merged)
        self._merged_generated_model2d.simulated_profiles[zvals_property_name] = simulated_profile_merged
    
    def simulate_profile_from_zvals_property_profile(self, main_property_name, zvals_property_name, 
                                            ignore_lithological_ids=['X'], 
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
            
            if (generated_model_instance.simulated_val_for_ignored_lit_property != self.spatial_simulator2d_instance.simulated_val_for_ignored_lit_property):
                raise ValueError(
                    f"Inconsistent simulated_val_for_ignored_lit_property: "
                    f"{generated_model_instance.simulated_val_for_ignored_lit_property} "
                    f"!= {self.spatial_simulator2d_instance.simulated_val_for_ignored_lit_property}"
                )
                
            simulated_profile = self.spatial_simulator2d_instance.simulate_profile_from_zvals_lit_profile(
                simulated_zvals_lit_profile, lit_domain,
                processed_property_dict, gwt_depth, warn_inconsistent_stdev = True,
                ignore_lithological_ids=ignore_lithological_ids,
                )
            
            simulated_profile = self.clip_simulated_profile(simulated_profile, min_val, max_val, generated_model_instance.simulated_val_for_ignored_lit_property, warn_cliping)      
            self._generated_model2d_set[set_name].simulated_profiles[main_property_name] = simulated_profile
            
            merged_lit_domain = self.merged_generated_model2d.lit_domain
            if order_id==0:
                simulated_profile_merged = None
                
            simulated_profile_merged = self.get_merged_simulated_profile(
                order_id, current_merged_lit_domain=merged_lit_domain, current_merged_simulated_profile=simulated_profile_merged,
                to_merge_lit_domain = lit_domain, to_merge_simulated_profile=simulated_profile, 
                simulated_val_for_ignored_lit_property=generated_model_instance.simulated_val_for_ignored_lit_property)
            
        # self.check_simulated_profile(simulated_profile_merged)
        simulated_profile_merged = self.clip_simulated_profile(simulated_profile_merged, 
                                                               min_val, max_val, generated_model_instance.simulated_val_for_ignored_lit_property, raise_error=True)      
        
        self._merged_generated_model2d.simulated_profiles[main_property_name] = simulated_profile_merged

    @staticmethod
    def clip_simulated_profile(simulated_profile, min_val=None, max_val=None, warn=False,
                            simulated_val_for_ignored_lit_property=-99999, raise_error=False):
        
        simulated_profile = np.array(simulated_profile)  # ensure NumPy array

        # Determine min and max if not provided
        valid_mask = simulated_profile != simulated_val_for_ignored_lit_property
        
        if min_val is not None:
            assert isinstance(min_val, (int, float)), f"min_val must be a float/int. Provided {type(min_val)} : {min_val}"
        else:
            min_val = np.min(simulated_profile[valid_mask])
            
        if max_val is not None:
            assert isinstance(max_val, (int, float)), f"max_val must be a float/int. Provided {type(max_val)} : {max_val}"
        else:
            max_val = np.max(simulated_profile[valid_mask])
        
        # Check values outside the clipping range (ignoring special values)
        values_outside_range = np.sum((simulated_profile[valid_mask] < min_val) | 
                                    (simulated_profile[valid_mask] > max_val))

        if values_outside_range > 0:
            msg = f"{values_outside_range} values were clipped to the range [{min_val}, {max_val}]. " \
                f"Actual Range [{np.min(simulated_profile[valid_mask])}, {np.max(simulated_profile[valid_mask])}]"
            if warn:
                warnings.warn("Warning: " + msg)
            if raise_error:
                raise ValueError(msg)
        
        # Clip only the valid values
        clipped_profile = simulated_profile.copy()
        clipped_profile[valid_mask] = np.clip(simulated_profile[valid_mask], min_val, max_val)
        
        return clipped_profile
        
    def simulate_property_profile(
        self, main_property_name, 
        ignore_lithological_ids=['X'], 
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
            ignore_lithological_ids=ignore_lithological_ids)
        
        self.simulate_profile_from_zvals_property_profile(
            main_property_name, zvals_property_name, 
            ignore_lithological_ids=ignore_lithological_ids, 
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
    
    @classmethod
    def from_config(cls, config_dict):
        return super().from_config(config_dict, read_only=False, check_merged=True)
        
def save_dict_to_hdf5(d, parent_group, compression_level=0):
    """
    Recursively saves a dictionary to an HDF5 group, with optional compression.

    Args:
        d (dict): Dictionary to save.
        parent_group (h5py.Group): HDF5 group to save into.
        compression_level (int): GZIP compression level (0–9). 0 disables compression.
    """
    for key, value in d.items():
        if isinstance(value, dict):
            # Create a subgroup for nested dictionaries
            group = parent_group.create_group(key)
            save_dict_to_hdf5(value, group, compression_level)
        else:
            if value is None:
                value = '__None__'
            # Save the value as a dataset
            # if key == 'lithological_matrix':
            #     value = convert_string_array_for_hdf5(value)
            #     dtype=h5py.string_dtype(encoding='utf-8')
            elif key in ("state", "inc") and not isinstance(value, dict):
                value=str(value)
                dtype=h5py.string_dtype("utf-8")
            else:
                dtype = None
            
            # Detect if value is array-like
            is_array = isinstance(value, (np.ndarray, list, tuple))
                                
            try:
                if is_array:
                    # Convert lists/tuples to np.ndarray
                    value_arr = np.array(value) if not isinstance(value, np.ndarray) else value
                    
                    if value_arr.dtype.kind in {"U", "S", "O"}:
                        dtype = h5py.string_dtype("utf-8")
                        value_arr = value_arr.astype(str)
                        
                    if value_arr.dtype.kind == "U":
                        value_arr = convert_string_array_for_hdf5(value_arr)
                        dtype = None  # bytes dtype inferred automatically

                        
                    if compression_level == 0:
                        parent_group.create_dataset(key, data=value_arr, dtype=dtype)
                    else:
                        parent_group.create_dataset(key, data=value_arr, dtype=dtype,
                                                    compression='gzip',
                                                    compression_opts=compression_level)
                else:
                    # Scalar values: no compression
                    parent_group.create_dataset(key, data=value)
            except TypeError as e:
                print(f"Warning: Could not save key '{key}' with value type {type(value)}. Error: {e}")

def convert_string_array_for_hdf5(string_array):
    shape = np.array(string_array).shape
    return np.array([s.encode('utf-8') for s in np.array(string_array).flatten()]).reshape(shape)#, np.array(string_array.shape)
