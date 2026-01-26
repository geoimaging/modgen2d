import numpy as np
import copy
import geomodgen2d.general_functions as f
from .a_base import LithologicalDomain2DReadOnly
from .a_from_interface import LithologicalDomain2D
from .a_from_obs2d import LithologicalDomain2DFromObstruction2D
from geomodgen2d.global_soil_interface_config import GlobalSoilInterfaceConfig

class LithologicalDomain2DCollection:
    """
    A class for generating and managing geotechnical lithological profiles in 2D.
    Supports interface-based and obstruction-based lithological domains, including
    merging, ordering, validation, and serialization.
    """
    def __init__(self, valid_feature_ids, interface_set_name:str="def"):
        """
        Initializes the LithologicalDomain2DCollection.

        Parameters
        ----------
        valid_feature_ids : list
            List of valid feature IDs for validation against lithological domains.
        interface_set_name : str, default "def"
            Name for the interface-based lithology set.
        """
        self.interface_config_revision_id = GlobalSoilInterfaceConfig.get_revision_id()
        self.interface_set_name = interface_set_name
        self.valid_feature_ids = valid_feature_ids
        self._gwt_depth = None
        
        self._lit_domain_set = {}
        self._merged_lit_domain = None
        self._invalid_set_names = ['merged', '']
        
        self._all_lit_ids = {}
        self._lit_id2material_dict = {}
        self._unique_code = 0
        self._locked = False
        self._read_only = True
    
    @property
    def all_lit_ids(self):
        """Returns a dictionary mapping lithology prefixes to all associated IDs."""
        return self._all_lit_ids
    
    @property
    def gwt_depth(self):
        """Returns the groundwater table (GWT) depth associated with the merged domain."""
        return self._gwt_depth
    
    @property
    def lit_domain_set(self):
        """Returns the set of all 'LithologicalDomain2D' instance in dictionary form with 'set name' as keys."""
        return self._lit_domain_set
    
    @property
    def merged_lit_domain(self):
        """Returns the merged 'LithologicalDomain2D' instance after locking."""
        return self._merged_lit_domain
    
    @property
    def unique_code(self):
        """Returns the unique code assigned to the locked collection."""
        return self._unique_code
    
    def unlock(self, delete_all_sets=False):
        """
        Unlocks the collection, optionally deleting all lithological domain sets.

        Parameters
        ----------
        delete_all_sets : bool, default False
            If True, removes all existing lithological domain sets.
        """
        if self._read_only:
            raise ValueError("This class is ReadOnly. Hence, cannot be unlocked.")
        self._merged_lit_domain = None
        
        if delete_all_sets:
            self._lit_domain_set = {}
            
        self._unique_code = 0
        self._locked = False
        
    def add_lithological_domain_from_soil_interface_config(self, lithological_domain_from_soil_interface_instance:LithologicalDomain2D):
        """
        Adds a lithological domain derived from the soil interface configuration.

        Parameters
        ----------
        lithological_domain_from_soil_interface_instance : LithologicalDomain2D
            Unmerged interface-based lithological domain to add.
        """
        if self._locked:
            raise SystemError("Operation cannot be performed because the object is locked.")
        
        if not isinstance(lithological_domain_from_soil_interface_instance, LithologicalDomain2D):
            raise TypeError("lithological_domain_from_soil_interface_instance must be an instance from LithologicalDomain2D")
        
        if lithological_domain_from_soil_interface_instance.lm_type != "from_interface_config":
            raise TypeError(f"lithological_domain_from_soil_interface_instance must be an lm type: 'from_interface_config'; must be unmerged lit_domain from interface config. Provided {lithological_domain_from_soil_interface_instance.lm_type}.")
        
        self.interface_config_revision_id = lithological_domain_from_soil_interface_instance.interface_config_revision_id
        self.__add_or_replace_lithological_domain(self.interface_set_name, lithological_domain_from_soil_interface_instance, 0, False)
        self._gwt_depth = lithological_domain_from_soil_interface_instance.gwt_depth
            
    def add_lithological_domain_from_obstruction2d(self, set_name:str, lithological_domain_from_obstruction2d_instance, lit_order=None):
        """
        Adds a lithological domain derived from a 2D obstruction.

        Parameters
        ----------
        set_name : str
            Name of the obstruction-based lithology set.
        lithological_domain_from_obstruction2d_instance : LithologicalDomain2DFromObstruction2D
            Obstruction-based lithological domain to add.
        lit_order : int or float, optional
            Order for merging. Must be positive.
        """
        if self._locked:
            raise SystemError("Operation cannot be performed because the object is locked.")
        
        if set_name == self.interface_set_name:
           raise ValueError(f"set_name: {set_name} is defined for interface. Cannot use for obstruction2ds.")       
        
        if lit_order is not None and lit_order<=0:
            raise ValueError(f"lit order must be positive number (>0) for lithological_domain_from_obstructions2D_instance.")
        
        if not isinstance(lithological_domain_from_obstruction2d_instance, LithologicalDomain2DFromObstruction2D):
            raise TypeError("lithological_domain_from_obstruction2d_instance must be an instance from LithologicalDomain2DFromObstruction2D")
        
        if set_name in self.lit_domain_set.keys():
            raise ValueError(f"set_name: {set_name} already defined. Remove that first if need to replace.")
        
        self.__add_or_replace_lithological_domain(set_name, lithological_domain_from_obstruction2d_instance, lit_order, allow_merged_lit=True)
    
    def delete_lithological_domain_from_obstruction2d(self, set_name:str):
        """
        Deletes a previously added obstruction-based lithological domain.

        Parameters
        ----------
        set_name : str
            Name of the set to delete.
        """
        if self._locked:
            raise SystemError("Operation cannot be performed because the object is locked.")
        
        if set_name==self.interface_set_name:
            raise ValueError(f"set_name is the defined interface_set_name : {set_name}. Cannot delete that. Use .refresh_interface_config() to refresh it.")

        if set_name not in self.lit_domain_set.keys():
            raise ValueError(f"set_name: {set_name} not created yet. Hence, unable to delete.")
        
        self._lit_domain_set.pop(set_name)
    
    def get_lit_orders(self, return_new_order=False):
        """
        Returns lithological orders for all sets.

        Parameters
        ----------
        return_new_order : bool, default False
            If True, returns reassigned sequential orders starting from 0.

        Returns
        -------
        dict
            Mapping from set_name to lith_order.
        """
        # Extract {set_name: lit_order}
        lit_orders = {
            set_name: val.lit_order
            for set_name, val in self.lit_domain_set.items()
        }

        # Stable sort by lit_order (ascending)
        ordered = sorted(lit_orders.items(), key=lambda x: x[1])

        # If requested, reassign new sequential order (0,1,2,...)
        if return_new_order:
            return {set_name: idx for idx, (set_name, _) in enumerate(ordered)}

        # Otherwise return the sorted original lit_order values
        return dict(ordered)

    def __add_or_replace_lithological_domain(self, set_name:str, lithological_domain_instance:LithologicalDomain2DReadOnly, lit_order=None, allow_merged_lit=False):
        """
        Adds a profile set with strict type validation.

        Parameters
        ----------
        set_name : str
            Name of the profile set.
        lithological_domain_instance : LithologicalDomain2DReadOnly
            Base lithological domain.
        allow_merged_lit : bool
            Whether merged lithology is allowed.
        """
        if self._locked:
            raise SystemError("Operation cannot be performed because the object is locked.")
        
        ## TypeError if not str
        ## TypeError if not list (1D of type LithologicalDomain2DReadOnly)
        ## TypeError if base_lithological_domain not None or instance of LithologicalDomain2DReadOnly
        # ---- Type checks ----
        # 1. set_name must be str
        if not isinstance(set_name, str):
            raise TypeError(
                f"set_name must be str, but got {type(set_name).__name__}"
            )

        # 2. allow_merged_lit must be bool
        if not isinstance(allow_merged_lit, bool):
            raise TypeError(
                f"allow_merged_lit must be bool, got {type(allow_merged_lit).__name__}"
            )
            
        # 3. Each element must be LithologicalDomain2DFromObstruction2D
        if not isinstance(lithological_domain_instance, LithologicalDomain2DReadOnly):
            raise TypeError(
                f"lithological_domain_from_obs_instance must be "
                f"LithologicalDomain2DReadOnly, got {type(lithological_domain_instance).__name__}"
            )
        
        if not allow_merged_lit:
            if lithological_domain_instance.merged_lit:
                raise ValueError(
                f"lithological_domain_from_obs_instance must NOT be merged lithological domain."
                f"Keep flag allow_merged_lit True, if want to allow merged lithological domain instance."
            )

        # 4. allow_merged_lit must be bool
        if not isinstance(allow_merged_lit, bool):
            raise TypeError(
                f"allow_merged_lit must be bool, got {type(allow_merged_lit).__name__}"
            )

        # 5. lit_order must be either None or a number.
        if lit_order is not None and not isinstance(lit_order, (int, float)):
            raise TypeError(
                f"lit_order must be None or a number (int or float). Got {type(lit_order).__name__}"
            )
            
        if lit_order is None:
            lit_order = self.__get_auto_lit_order()
            
        if set_name in self._invalid_set_names:
            raise ValueError(f"set_name cannot be either 'merged' (Reserved), or ''. Provided {set_name}")
        
        if set_name == self.interface_set_name:
            if lit_order!=0:
                raise ValueError(f"For interface lithological domain, lit order Must be 0. Procided {lit_order}")
        else:
            if lit_order<=0:
                raise ValueError(f"lit order must be positive number (>0) for lithological_domain_from_obstructions2D_instance.")
        
        if not GlobalSoilInterfaceConfig.get_config_status(self.interface_config_revision_id):
            lithological_domain_instance.refresh() #Compute for new surface
            
        lit_dict = lithological_domain_instance.get_feature_id_and_lit_val_from_lithological_matrix()

        feature_ids = lit_dict.keys()
        invalid = [fid for fid in feature_ids if fid not in self.valid_feature_ids]

        if invalid:
            raise ValueError(
                f"The following feature_ids are invalid (not in MainPropertiesConfig): {invalid}"
            )
    
        lithological_domain_instance.lit_order = lit_order
        self._lit_domain_set[set_name] = lithological_domain_instance
        
    def __get_auto_lit_order(self):
        lit_orders=[]
        for _,val in self.lit_domain_set.items():
            lit_orders.append(val.lit_order)
        
        return float(np.max(lit_orders)+1)

    def __reorder_lit_order(self):
        
        ordered = self.get_lit_orders(return_new_order=True)
        # gwt_depth_list = []
        
        for set_name, lit_id in ordered.items():
            self._lit_domain_set[set_name].lit_order = lit_id
            # gwt_depth_list.append(self._generated_model_set[set_name].gwt_depth)
            
        # Rebuild dictionary in ascending lit_order
        sorted_keys = sorted(ordered, key=ordered.get)  # sorted by new lit_order
        self._lit_domain_set = {
            key: self.lit_domain_set[key]
            for key in sorted_keys
        }
        
        return ordered
        
    @staticmethod
    def __merge_lit_dicts(list_of_dicts):
        merged = {}

        for d in list_of_dicts:
            for key, vals in d.items():
                if key not in merged:
                    merged[key] = set(vals)
                else:
                    merged[key].update(vals)

        # Convert sets → sorted lists
        return {k: sorted(list(v)) for k, v in merged.items()}

    def __get_merged_set(self):
        # Step 1: Merge the lithological domain and create a PropertyProfileEach for merged.
        #          Also get all_lit_ids
        ordered = self.__reorder_lit_order()
        merged_lit_domain, merged_all_lit_ids, gwt = self.get_merged_lit_domain(ordered, self.lit_domain_set, self.valid_feature_ids)
        self._gwt_depth = gwt
        self._merged_lit_domain=merged_lit_domain
        self._all_lit_ids = merged_all_lit_ids
        
    @staticmethod
    def get_merged_lit_domain(ordered, lit_domain_set, valid_feature_ids):
        merged_all_lit_ids = {}
        for i, set_name in enumerate(ordered.keys()):
            # Merging
            lithological_domain_instance = copy.deepcopy(lit_domain_set[set_name])
        
            if not isinstance(lithological_domain_instance, LithologicalDomain2DReadOnly):
                raise TypeError(f"The values of all generated_profiles_set must be LithologicalDomain2DReadOnly. Not the case for {set_name}")
            
            if i!=lithological_domain_instance.lit_order:
                raise SystemError("Sets must be in order. It should have been in order auto. Check codes.")
            
            lithological_domain_instance.check_shape()
            
            if i == 0:
                if lithological_domain_instance.check_for_Xs(lithological_domain_instance.lithological_matrix):
                    raise TypeError(f"The first lit_domain set in generated_profiles_set must be LithologicalDomain2D with no 'X's. Not the case the one in first set : {set_name}")
                
                if not isinstance(lithological_domain_instance, LithologicalDomain2D):
                    config = lithological_domain_instance.get_config
                    lithological_domain_instance = LithologicalDomain2D.from_config(config)
                
                merged_lit_domain = lithological_domain_instance
                gwt = lithological_domain_instance.gwt_depth
            else:
                # Perform nearest interp only
                merged_lit_domain = merged_lit_domain.return_merged_lithological_domain([lithological_domain_instance])
        
            # Getting all_lit_dicts
            lit_dict = lithological_domain_instance.get_feature_id_and_lit_val_from_lithological_matrix()
            
            feature_ids = lit_dict.keys()
            invalid = [fid for fid in feature_ids if fid not in valid_feature_ids]

            if invalid:
                raise ValueError(
                    f"The following feature_ids are invalid (not in MainPropertiesConfig): {invalid}"
                )
            merged_all_lit_ids = LithologicalDomain2DCollection.__merge_lit_dicts([merged_all_lit_ids, lit_dict])
        merged_lit_domain.lit_order=-1
        merged_lit_domain.check_shape()
        
        merged_lit_dict = merged_lit_domain.get_feature_id_and_lit_val_from_lithological_matrix()
          
        #Check : all_lit_ids makes sense
        for prefix, vals in merged_lit_dict.items():
            if prefix not in merged_all_lit_ids:
                raise ValueError(
                    f"Prefix '{prefix}' found in merged profile but not in all_lit_ids."
                )
            merged_set = set(vals)
            all_set     = set(merged_all_lit_ids[prefix])

            if not merged_set.issubset(all_set):
                missing = merged_set - all_set
                raise ValueError(
                    f"Merged profile contains lit values {missing} for prefix '{prefix}', "
                    f"which are not found in all_lit_ids."
                )
        return merged_lit_domain, merged_all_lit_ids, gwt

    def lock(self):
        """
        Locks the collection by merging all lithological domains and generating a unique code.
        After locking, no modifications can be made to the collection.
        """
        self.__get_merged_set()
        self._unique_code = np.random.randint(1, 10**18, dtype=np.int64)
        self._locked = True
        #self.check()

    @property
    def get_config(self):
        """
        Returns a serializable dictionary representing the current configuration
        of the LithologicalDomain2DCollection.
        """
        self_config = {}
        self_config['_all_lit_ids'] = self._all_lit_ids
        self_config['_gwt_depth'] = self._gwt_depth
        self_config['_invalid_set_names'] = self._invalid_set_names

        self_config['_lit_domain_set'] = {}
        for key, val in self._lit_domain_set.items():
            self_config['_lit_domain_set'][key] = val.get_config
            
        self_config['_lit_id2material_dict'] = self._lit_id2material_dict
        self_config['_locked'] = self._locked
        self_config['_merged_lit_domain'] = self._merged_lit_domain.get_config
        self_config['_unique_code'] = self._unique_code
        self_config['interface_config_revision_id'] = self.interface_config_revision_id
        self_config['interface_set_name'] = self.interface_set_name
        self_config['valid_feature_ids'] = self.valid_feature_ids
        return self_config

    @classmethod
    def from_config(cls, config_dict, read_only=False):
        """
        Creates a LithologicalDomain2DCollection instance from a configuration dictionary.

        Parameters
        ----------
        config_dict : dict
            Serialized configuration dictionary from get_config.
        read_only : bool, default False
            If True, returns read-only LithologicalDomain2DReadOnly instances.

        Returns
        -------
        LithologicalDomain2DCollection
            Restored collection object.
        """
        if not isinstance(config_dict, dict):
            raise TypeError("Expected a dictionary.")
        try:
            obj = cls.__new__(cls) 
            
            obj._all_lit_ids = config_dict['_all_lit_ids']
            obj._gwt_depth = config_dict['_gwt_depth']
            obj._invalid_set_names = config_dict['_invalid_set_names']
            
            obj._lit_domain_set = {}
            for key, val in config_dict['_lit_domain_set'].items():
                if read_only:
                    obj._lit_domain_set[key] = LithologicalDomain2DReadOnly.from_config(val)
                else:
                    lm_type = val['lm_type']
                    if lm_type.startswith("from_interface_config"):
                        obj._lit_domain_set[key] = LithologicalDomain2D.from_config(val)
                    else:
                        obj._lit_domain_set[key] = LithologicalDomain2DFromObstruction2D.from_config(val)
                            
            obj._lit_id2material_dict = config_dict['_lit_id2material_dict']
            obj._locked = config_dict['_locked']
            
            if config_dict['_merged_lit_domain'] is None:
                obj._merged_lit_domain = config_dict['_merged_lit_domain']
            else:
                if read_only:
                    obj._merged_lit_domain = LithologicalDomain2DReadOnly.from_config(config_dict['_merged_lit_domain'])
                else:
                    obj._merged_lit_domain = LithologicalDomain2D.from_config(config_dict['_merged_lit_domain'])
                
            obj._unique_code = config_dict['_unique_code']

            obj.interface_config_revision_id = config_dict['interface_config_revision_id']
            obj.interface_set_name = config_dict['interface_set_name']
            obj.valid_feature_ids = config_dict['valid_feature_ids']
            return obj

        except (KeyError, TypeError) as e:
            raise ValueError(f"Invalid config dictionary: {e}")
        
    def __eq__(self, other):
        """
        Checks equality between two LithologicalDomain2DCollection objects.

        Parameters
        ----------
        other : LithologicalDomain2DCollection
            Another collection to compare with.

        Returns
        -------
        bool
            True if the collections are equivalent, False otherwise.
        """
        if not isinstance(other, LithologicalDomain2DCollection):
            return NotImplemented
        
        # units_check = self.units_config == other.units_config
        # spans_check = np.allclose(self._spans_in_domain_len_units, other._spans_in_domain_len_units)
        # dhs_check = np.allclose(self._dhs_in_domain_len_units, other._dhs_in_domain_len_units)
        # return (
        #     units_check
        #     and spans_check
        #     and dhs_check
        # )
        
#GWT Handling in modelgenerator remaining
