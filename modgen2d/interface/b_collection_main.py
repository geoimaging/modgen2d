"""
This module provides classes for managing and generating 2D spatial profiles
of geotechnical properties over lithological domains.

This is the main file of modgen2d that holds all the model objects, save and load them.
"""
import h5py
import warnings
# from IPython.display import clear_output
import numpy as np
import modgen2d.general_functions as f
from modgen2d.spatial_simulator2d import CovarianceDecompositionSimulator, SpatialSimulator2D    
from modgen2d.generated_model2d import GeneratedModel2D
from modgen2d.lithological_domain2d import LithologicalDomain2DCollection, LithologicalDomain2DReadOnly
from modgen2d.main_properties import MainPropertiesConfig
from modgen2d.global_soil_interface_config import GlobalSoilInterfaceConfig
from modgen2d.metadata import __version__

class GeneratedProfileCollection2DReadOnly:
    """
    Read-only collection of 2D generated property profiles over lithological domains. 
    Accessible via: `modgen2d.GeneratedProfileCollection2DReadOnly`.
    """
    def __init__(self, main_properties_config_instance: MainPropertiesConfig, lithological_domain2d_collection: LithologicalDomain2DCollection, spatial_simulator2d_instance:SpatialSimulator2D):
        """
        Initialize the 'GeneratedProfileCollection2DReadOnly' object.

        Parameters
        ----------
        main_properties_config_instance : MainPropertiesConfig
            Locked instance containing sampled property definitions.
        lithological_domain2d_collection : LithologicalDomain2DCollection
            Locked collection of lithological domains.
        spatial_simulator2d_instance : SpatialSimulator2D
            Simulator instance used for generating spatially correlated profiles.

        Raises
        ------
        TypeError
            If main_properties_config_instance or lithological_domain2d_collection are not locked.
        """
        if not main_properties_config_instance._locked:
            raise TypeError("main_properties_config_instance is not locked yet. Use .lock_and_generate_sample_properties first.")
        
        if not lithological_domain2d_collection._locked:
            raise TypeError("lithological_domain2d_collection is not locked yet. Use .lock first.")
        
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
        """List of generated property names."""
        return self._generated_properties_list
    
    @property
    def sampled_properties(self):
        """List of generated property names."""
        return self._sampled_properties
    
    @property
    def lit_id2material_dict(self):
        """Dictionary of sampled property definitions."""
        return self._lit_id2material_dict
    
    @property
    def spatial_simulator2d_instance(self):
        """Get the 'SpatialSimulator2D' instance used for simulations."""
        return self._spatial_simulator2d_instance

    def change_spatial_simulator_type(self, new_simulator_class):
        """
        Change the type of the spatial simulator.

        Parameters
        ----------
        new_simulator_class : type
            New class to replace the current spatial simulator.
        """
        self._spatial_simulator2d_instance.change_spatial_simulator_type(new_simulator_class)
        
    @property
    def main_properties_unique_code(self):
        """Unique code of the main property configuration."""
        return self._main_properties_unique_code
    
    def get_generated_model2d(self, set_name):
        if set_name is None:
            return self.merged_generated_model2d
        return self._generated_model2d_set[set_name]
    
    @property
    def generated_model2d_set(self):
        """Dictionary mapping set names to 'GeneratedModel2D' instances."""
        return self._generated_model2d_set
    
    @property
    def merged_generated_model2d(self):
        """Merged 'GeneratedModel2D' object across all sets."""
        return self._merged_generated_model2d
    
    @property
    def get_simulated_profiles(self, set_name):
        """Get simulated profiles dictionary for a given set name."""
        if set_name is None:
            return self.merged_generated_model2d.simulated_profiles
        return self.generated_model2d_set[set_name].simulated_profiles

    @property
    def get_simulated_properties(self):
        """Get keys of the simulated properties (from first set)."""
        keys = self.generated_model2d_set.keys()
        first_key = next(iter(keys), None)

        if first_key is None:
            return None

        return self.generated_model2d_set[first_key].simulated_profiles.keys()
    
    def check(self):
        """
        Validates the generated profiles.

        Checks performed:
        - GWT depth consistency across sets
        - Simulated value consistency for ignored lithological IDs
        - All simulated property keys match across generated sets
        - Each `GeneratedModel2D` passes its internal check
        """
        simulated_val_for_ignored_lit_property = self.spatial_simulator2d_instance.simulated_val_for_ignored_lit_property
        simulated_properties = self.get_simulated_properties
        
        # Check order_no
        ordered_set_names_dict, _ = self.get_ordered_set_names_dict()
        lit_orders = list(ordered_set_names_dict.keys())

        n = len(lit_orders)
        expected = set(range(n))
        if set(lit_orders) != expected:
            raise ValueError(
                f"Invalid lit_orders: expected {sorted(expected)}, "
                f"got {sorted(lit_orders)}"
            )
        
        first = True
        for set_name, gen_model2d in self._generated_model2d_set.items():
            if first:
                gwt_depth = gen_model2d.gwt_depth
            else:
                if gwt_depth != gen_model2d.gwt_depth:
                    raise ValueError(
                        f"Inconsistent gwt_depth across generated models: "
                        f"{gwt_depth} != {gen_model2d.gwt_depth}"
                    )
            
            if simulated_val_for_ignored_lit_property != gen_model2d.simulated_val_for_ignored_lit_property:
                raise ValueError(
                        f"simulated_val_for_ignored_lit_property mismatch for set '{set_name}'. "
                        f"Expected: From spatial simulator: {simulated_val_for_ignored_lit_property}, "
                        f"found {gen_model2d.simulated_val_for_ignored_lit_property}."
                    )
                
            simulated_properties_each = set(self.generated_model2d_set[set_name].simulated_profiles.keys())
            if simulated_properties_each != simulated_properties:
                raise ValueError(
                    f"Simulated properties mismatch for set '{set_name}'. "
                    f"Expected {sorted(simulated_properties)}, "
                    f"found {sorted(simulated_properties_each)}."
                )
                
            gen_model2d.check(ignore_lithological_ids=['X'], allow_ignored_lit_property=gen_model2d.lit_order!=0)
            
        self._merged_generated_model2d.check(ignore_lithological_ids=['X'], allow_ignored_lit_property=False)
        
    def get_ordered_set_names_dict(self):
        """
        Returns an ordered mapping of lithological set orders to set names.

        Returns
        -------
        tuple
            ordered_set_names_dict : dict
                Mapping of lithological order (int) → set name (str), sorted by order.
            lit_domain_set : dict
                Mapping of set names to their `LithologicalDomain2D` instances.
        """
        # Get ordered and lit_domain_set
        ordered_set_names_dict = {}
        lit_domain_set = {}
        
        for set_name, gen_model2d in self._generated_model2d_set.items():
            lit_domain = gen_model2d.lit_domain
            lit_order = gen_model2d.lit_order
            
            # Enforce uniqueness
            if lit_order in ordered_set_names_dict:
                raise ValueError(f"Duplicate lit_order detected: {lit_order}")
            if set_name in ordered_set_names_dict.values():
                raise ValueError(f"Duplicate set_name detected: {set_name}")
            
            ordered_set_names_dict[lit_order] = set_name
            lit_domain_set[set_name] = lit_domain
                
        # Sort by lit_order and convert back to dict
        ordered_set_names_dict = dict(
            sorted(ordered_set_names_dict.items(), key=lambda x: x[0])
        )
        
        return ordered_set_names_dict, lit_domain_set
    
    # To do: get merged and check if asme with the one generateed. If not woo woo!!! (MAYBE LATER)

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
        """
        Reconstructs a read-only collection from a configuration dictionary.

        Parameters
        ----------
        config_dict : dict
            Configuration dictionary obtained from `.get_config`.
        read_only : bool, optional
            Whether to enforce read-only mode. Default False.
        check_merged : bool, optional
            Whether to check merged profile consistency. Default False.

        Returns
        -------
        GeneratedProfileCollection2DReadOnly
            The reconstructed read-only profile collection.
        """
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

    def save_to_hdf5(self, file_name, hdf5_compression_level=6):
        """
        Save the full profile collection to an HDF5 file.

        Parameters
        ----------
        file_name : str
            File path to save the HDF5 file.
        hdf5_compression_level : int, optional
            GZIP compression level (0–9), default is 6 (0 means no compression).
        """
        self.check()
        to_save_dict = {
            'modgen2d_version': __version__,
            'save_read_only': False,
            'global_interface_config': GlobalSoilInterfaceConfig.get_config,
            'gen_model_2d_collection': self.get_config
        }
        
        with h5py.File(file_name, 'w') as hf:
            save_dict_to_hdf5(to_save_dict, hf, compression_level=hdf5_compression_level)
            
        print(f"Data saved to {file_name}")
        
    def save_to_hdf5_read_only(self, file_name, save_merged_only=True, save_interface=False, hdf5_compression_level=6):
        """
        Save a read-only version of the profile collection to HDF5.

        Parameters
        ----------
        file_name : str
            Path of the HDF5 file to save.
        save_merged_only : bool, optional
            Save only the merged model, default True.
        save_interface : bool, optional
            Include global interface configuration, default False.
        hdf5_compression_level : int, optional
            GZIP compression level (0–9), default is 6 (0 means no compression).
        """
        self.check()
        state_dict = self.get_config.copy()
        if save_merged_only:
            state_dict.pop('_generated_model2d_set', None)
                
        to_save_dict = {
            'modgen2d_version': __version__,
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
        """
        Save the profile collection as NumPy arrays in HDF5 format.

        Parameters
        ----------
        file_name : str
            File path to save.
        save_merged_only : bool, optional
            Save only merged model, default True.
        save_interface : bool, optional
            Save interface matrix, default False.
        save_lithological_domain : bool, optional
            Include lithological domain data, default False.
        save_properties_metadata : bool, optional
            Include property metadata, default False.
        hdf5_compression_level : int, optional
            GZIP compression level (0–9), default is 6 (0 means no compression).
        """
        self.check()
        with h5py.File(file_name, 'w') as hf:
            to_save_dict = {
                'modgen2d_version': __version__,
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
    """
    Editable subclass of `GeneratedProfileCollection2DReadOnly` for generating
    and managing 2D spatial property profiles.
    Accessible via: `modgen2d.GeneratedProfileCollection2D`.
    """
    def __init__(self, main_properties_config_instance: MainPropertiesConfig, lithological_domain2d_collection: LithologicalDomain2DCollection, spatial_simulator2d_instance:SpatialSimulator2D):
        """
        Delete a simulated property profile across all sets and merged model.

        Parameters
        ----------
        property_name : str
            Name of the property to delete.

        Raises
        ------
        ValueError
            If the property does not exist.
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
        """
        Merge simulated profiles from a new set into the current merged profile.

        Parameters
        ----------
        current_merged_lit_domain : LithologicalDomain2DReadOnly
            Current merged lithological domain.
        current_merged_simulated_profile : np.ndarray or None
            Current merged profile array.
        to_merge_lit_domain : LithologicalDomain2DReadOnly
            Lithological domain of the new set.
        to_merge_simulated_profile : np.ndarray
            Simulated profile of the new set.
        simulated_val_for_ignored_lit_property : float
            Value for ignored lithological IDs.

        Returns
        -------
        np.ndarray
            Updated merged simulated profile.
        """
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
            simulated_profile_merged = current_merged_simulated_profile
        else:
            # Masks to identify non-'none' values
            mask_other_non_none = ~np.isin(to_merge_simulated_profile_remeshed, simulated_val_for_ignored_lit_property)
            simulated_profile_merged = np.where(mask_other_non_none, to_merge_simulated_profile_remeshed, current_merged_simulated_profile)  # (replace it with that of other class even if merged already have values, i.e prioritize other class)
        
        return simulated_profile_merged
   
    def simulate_zvals_property_profile(self, zvals_property_name,  
                                            generate_non_spatial_profile=False, 
                                            ignore_lithological_ids=['X']):
        """
        Generate a z-values property profile for all sets and merged model.

        Parameters
        ----------
        zvals_property_name : str
            Name of the z-values property to generate.
        generate_non_spatial_profile : bool, optional
            Whether to generate a non-spatial profile, default False.
        ignore_lithological_ids : list of str, optional
            Lithological IDs to ignore, default ['X'].

        Raises
        ------
        ValueError
            If property name conflicts with sampled or already simulated properties.
        """
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
                current_merged_lit_domain=merged_lit_domain, current_merged_simulated_profile=simulated_profile_merged,
                to_merge_lit_domain = lit_domain, to_merge_simulated_profile=simulated_profile, 
                simulated_val_for_ignored_lit_property=generated_model_instance.simulated_val_for_ignored_lit_property
                )
            
        # self.check_simulated_profile(simulated_profile_merged)
        self._merged_generated_model2d.simulated_profiles[zvals_property_name] = simulated_profile_merged
    
    def simulate_profile_from_zvals_property_profile(self, main_property_name, zvals_property_name, 
                                            ignore_lithological_ids=['X'], 
                                            min_val = None, max_val = None, warn_cliping = False):
        """
        Generate main property profile from pre-generated z-values profile.

        Parameters
        ----------
        main_property_name : str
            Property name to generate.
        zvals_property_name : str
            Name of the pre-generated z-values profile.
        ignore_lithological_ids : list of str, optional
            Lithological IDs to ignore, default ['X'].
        min_val : float or int, optional
            Minimum value for clipping.
        max_val : float or int, optional
            Maximum value for clipping.
        warn_cliping : bool, optional
            Warn on clipping, default False.

        Raises
        ------
        ValueError
            If property constraints are violated.
        """
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
                current_merged_lit_domain=merged_lit_domain, current_merged_simulated_profile=simulated_profile_merged,
                to_merge_lit_domain = lit_domain, to_merge_simulated_profile=simulated_profile, 
                simulated_val_for_ignored_lit_property=generated_model_instance.simulated_val_for_ignored_lit_property)
            
        # self.check_simulated_profile(simulated_profile_merged)
        simulated_profile_merged = self.clip_simulated_profile(simulated_profile_merged, 
                                                               min_val, max_val, generated_model_instance.simulated_val_for_ignored_lit_property, raise_error=True)      
        
        self._merged_generated_model2d.simulated_profiles[main_property_name] = simulated_profile_merged

    @staticmethod
    def clip_simulated_profile(simulated_profile, min_val=None, max_val=None, warn=False,
                            simulated_val_for_ignored_lit_property=-99999, raise_error=False):
        """
        Clip profile values to a specified range, ignoring lithological placeholders.

        Parameters
        ----------
        simulated_profile : np.ndarray
            The simulated profile to clip.
        min_val : float or int, optional
            Minimum value. Computed from the profile if not provided.
        max_val : float or int, optional
            Maximum value. Computed from the profile if not provided.
        warn : bool, optional
            Whether to issue a warning for clipped values. Default False.
        simulated_val_for_ignored_lit_property : float, optional
            Value representing ignored lithological IDs. Default -99999.
        raise_error : bool, optional
            Raise ValueError if any values are clipped. Default False.

        Returns
        -------
        np.ndarray
            The clipped profile array.
        """
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
        Generate a spatial profile for a specified property.

        This method creates an intermediate z-values profile, converts it
        into the target property, merges across lithological sets, clips
        values, and removes the temporary z-values.

        Parameters
        ----------
        main_property_name : str
            Name of the property to generate.
        ignore_lithological_ids : list of str, optional
            Lithological IDs to ignore during simulation. Values are set to -99999. Default ['X'].
        min_val : float or int, optional
            Minimum value for clipping the profile.
        max_val : float or int, optional
            Maximum value for clipping the profile.
        warn_cliping : bool, optional
            Whether to warn if values are clipped. Default False.
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
    
    @classmethod
    def from_config(cls, config_dict):
        return super().from_config(config_dict, read_only=False, check_merged=True)
        
def save_dict_to_hdf5(d, parent_group, compression_level=6):
    """
    Recursively save a dictionary to an HDF5 group.

    Parameters
    ----------
    d : dict
        Dictionary to save.
    parent_group : h5py.Group
        HDF5 group to save into.
    compression_level : int, optional
        GZIP compression level (0–9), default 0.
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
    """
    Convert a numpy string array to byte-encoded format for HDF5 storage.

    Parameters
    ----------
    string_array : np.ndarray
        Array of strings.

    Returns
    -------
    np.ndarray
        Byte-encoded array suitable for HDF5.
    """
    shape = np.array(string_array).shape
    return np.array([s.encode('utf-8') for s in np.array(string_array).flatten()]).reshape(shape)#, np.array(string_array.shape)
