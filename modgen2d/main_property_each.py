# Sanish Bhochhibhoya
# Apr 12: Changed the Structure of properties. 
#   Was: Main Property >> Feature ID >> wet/dry/both >> Dict of Random Generators or Random Generator
#   Now: Main Property >> Feature ID >> Dictionary of material types as keys >> Wet/dry/both >> Random Generator Instance.

from modgen2d.property_distribution import PropertyDistribution
from modgen2d.features_config import FeaturesConfig
import modgen2d.general_functions as f
import pprint

class MainProperty:
    __slots__ = ['_locked', '_main_property_name', '_features_config', '_layer0_flag', '_main_property_dist', '_description']
    """
    Manages a primary material property across all features and materials.

    This class organizes property distributions by feature ID, material type, and wet/dry state.
    """
    def __init__(self, main_property_name, features_config:FeaturesConfig, layer0_flag:bool=False, description=''):
        """
        Initializes the 'MainProperty' class. 
        
        Note: Must have a layer0 (int/floats, optional): Default property values for air/water(layer0). (Default: None; i.e. Error generated when this value is ever to be used.)

        Parameters
        ----------
        main_property_name : str
            Name of the property being generated.
        features_config : FeaturesConfig
            Feature and material configuration.
        layer0_flag : bool, optional
            If True, allows definition of a default 'layer0' property
            for air or water.
        description : str, optional
            Description of the main property.
        
        Notes
        -----
        Hierarchy:
        Main_Properties_Set >> MainProperty instances >> Feature_ID >> material >> wet/dry_properties (Property instance.)
        
        """
        self._locked = False  #Locked to no edit (IE Checked.)
        
        # Ensure name is string.
        if not isinstance(main_property_name, str):
            raise ValueError("main_property_name must be a string")
        self._main_property_name = main_property_name

        features_config.check() #already made sure def exist in feature ids, layer0 does not exist in any material name

        self._features_config = features_config
        self._layer0_flag = layer0_flag
        # ALL VERIFIED. Now formatting all_properties instance accordingly.
        prop_dict = {}
        
        for id_ in features_config.get_feature_ids():
            # if id_ == 'def':
            #     prop_dict[id_] = {'layer0':layer0} #Initialized as same property for wet and dry case (GWT)
            # else:                
            prop_dict[id_] = {}
            for mat_prop in features_config.get_material_types(id_):
                prop_dict[id_][mat_prop] = {}
        
        if layer0_flag:
            prop_dict['def']['layer0'] = {}

        self._main_property_dist = f.FixedKeyDict(prop_dict)
        self._description = description
        
    @property
    def main_property_name(self):
        return self._main_property_name
    
    @property
    def features_config(self):
        return self._features_config
    
    @property
    def layer0_flag(self):
        return self._layer0_flag
    
    @property
    def main_property_dist(self):
        return self._main_property_dist
    
    @property
    def description(self):
        return self._description
    
    @property
    def locked(self):
        return self._locked
    
    def unlock(self):
        self._locked = False
        
    def print(self):
        """Prints the main properties, subdivisions with IDs, and all stored properties."""
        print(f"Property Name: {self.main_property_name}")
        if self.description != '':
            print(f"Description: {self.description}")
        print(f"All Feature IDs: {self.features_config.get_feature_ids()}")
        print("Properties:")
        pprint.pp(self.main_property_dist)
    
    def add_material_property_of_feature(self, feature_id, material_name, property_distribution_instance:PropertyDistribution, property_distribution_instance_if_dry:PropertyDistribution=None):
        """
        Assign a property distribution to a material within a feature.

        Parameters
        ----------
        feature_id : str
            Feature identifier.
        material_name : str
            Material name.
        property_distribution_instance : PropertyDistribution
            Property distribution for wet or both conditions.
        property_distribution_instance_if_dry : PropertyDistribution or None, optional
            Property distribution for dry conditions. If None, the same
            distribution is used for both wet and dry.

        """
        if self.locked:
            raise RuntimeError(f"The MainProperty instance is locked. If need to unlock, run mp_var.unlock() first. Issue in main property {self.main_property_name}.")
        
        if feature_id not in self.main_property_dist.keys():
            raise KeyError(f"In main property {self.main_property_name}: feature_id ({feature_id}) is not in main_property_dist.keys")

        if material_name == 'layer0' and feature_id != 'def':
            raise ValueError(f"'layer0' is a reserved name for feature_id: 'def' only. Issue in main property {self.main_property_name}, feature_id '{feature_id}', and material {material_name}.")
        
        material_property_dict:dict = self.main_property_dist[feature_id]       
        if material_name not in material_property_dict.keys():
            raise ValueError(f"Material ({material_name}) does not exist in provided features config (or is not 'layer0' for feature: 'def'). Issue in main property {self.main_property_name}, for feature_id ({feature_id}).")
        
        if material_property_dict[material_name]:
            raise ValueError (f"In main property {self.main_property_name}: Properties for {material_name} for feature_id '{feature_id}', already defined.")
        
        if not isinstance(property_distribution_instance, PropertyDistribution):
            raise TypeError(f"Property_instance must be an instance of the PropertyDistribution class. Issue in main property {self.main_property_name}, feature_id '{feature_id}', and material {material_name}.")

        if property_distribution_instance.property_name != self.main_property_name:
            raise ValueError(f"Property name in distribution_instance {property_distribution_instance.property_name} does not match with the main property name. Issue in main property {self.main_property_name}, feature_id '{feature_id}', and material {material_name}.")
        property_distribution_instance.check_class()

        if property_distribution_instance_if_dry is not None:
            if property_distribution_instance_if_dry.property_name != self.main_property_name:
                raise ValueError(f"Property name in distribution_instance_if_dry {property_distribution_instance_if_dry.property_name} does not match with the main property name. Issue in main property {self.main_property_name}, feature_id '{feature_id}', and material {material_name}.")
            property_distribution_instance_if_dry.check_class()
        
            self._main_property_dist[feature_id][material_name] = f.FixedKeyDict({'wet': property_distribution_instance,
                                                                         'dry': property_distribution_instance_if_dry})
        else:
            self._main_property_dist[feature_id][material_name] = f.FixedKeyDict({'both': property_distribution_instance})     
            # Should raise error as main_prop_dist was fixedkeydist.
            # TODO: rewrite tests. But first write for FeaturesConfig
            
    def _check_material_property_distributions_in_each_feature_id(self, feature_id):
        material_property_dict:dict = self.main_property_dist[feature_id]
        material_names = list(material_property_dict.keys())

        # Cannot empty
        if not material_property_dict:
            raise ValueError(f"Material property dictionary is empty. Issue in main property {self.main_property_name}, and feature_id '{feature_id}'.")

        # must be all numbers OR all strings
        if not (
            all(isinstance(x, str) for x in material_names)
        ):
            raise ValueError(f"Material_names must be either all strings. Issue in main property {self.main_property_name}, and feature_id '{feature_id}'.")
        
        # ---- Validate each entry ----
        for material_name, props in material_property_dict.items():
            # special case
            if material_name == 'layer0':
                if self.layer0_flag:
                    if feature_id != 'def':
                        raise ValueError(f"'layer0' is a reserved name for feature_id: 'def' only. Issue in main property {self.main_property_name}, and feature_id '{feature_id}'.")
                else:
                    raise ValueError(f"'layer0_flag' is flagged False. Hence, cannot be defined. Issue in main property {self.main_property_name}, and feature_id '{feature_id}'.")

            # valid keys
            keys = set(props.keys())
            if keys not in ({"wet", "dry"}, {"both"}):
                raise ValueError(
                    f"Material must have keys ('wet','dry') or ('both'). Seems property not added in main property {self.main_property_name}, feature_id '{feature_id}', and material: {material_name}."
                )

            # check the generator instances
            for property_inst in props.values():
                if self.main_property_name != property_inst.property_name:
                    raise ValueError(f"Property name in distribution_instance {property_inst.property_name} does not match with the main property name. Issue in main property {self.main_property_name}, feature_id '{feature_id}', and material: {material_name}.")
                property_inst.check_class()
                
    def lock_and_check(self):
        """
        Validate all property distributions and lock the instance.

        After locking, no further modifications are allowed.
        """
        self.features_config.check()
        for id_ in self.main_property_dist.keys():
            material_list1= list(self.main_property_dist[id_].keys())
            if 'layer0' in material_list1:
                material_list1.remove('layer0')
            material_list2= self.features_config.get_material_types(id_)
            
            # Make sure the two material lists match
            if set(material_list1) != set(material_list2):
                raise ValueError(
                    f"Material list mismatch for feature_id '{id_}': "
                    f"{material_list1} != {material_list2}. Issue in main property {self.main_property_name}."
                )
            
            self._check_material_property_distributions_in_each_feature_id(feature_id=id_)
                
        self._locked = True

    #@property
    def generate_sample_dict(self, feature_id, feature_material_name):
        """
        Generate a sampled property dictionary for a material.
    
        Parameters
        ----------
        feature_id : str
            Feature identifier.
        feature_material_name : str
            Material name.

        Returns
        -------
        FixedKeyDict
            Dictionary containing sampled mean and variability values.

        Raises
        ------
        RuntimeError
            If the instance is not locked.
        ValueError
            If the feature or material is invalid.
        """
        if not self.locked:
            raise RuntimeError("Cannot generate dictionary until the MainProperty instance is locked (checked). Need to run mp_var.lock_and_check()")
        # self.lock_and_check() 
        
        if feature_id in ['']:
            raise ValueError("Invalid feature id")
        
        if feature_id not in self.main_property_dist.keys():
            raise ValueError(f'Provided feature id {feature_id} not in properties.keys()')
        
        features_in_dict = list(self.main_property_dist[feature_id].keys())
        if feature_material_name not in features_in_dict:
            raise ValueError(f"feature_id ({feature_material_name}) not in main_property_distribution's features. features available: {features_in_dict}")
        
        sample_dict_prop_id = _get_random_generated_sample(self.main_property_dist[feature_id][feature_material_name])
         
        return f.FixedKeyDict(sample_dict_prop_id)
    
def _get_random_generated_sample(material_property_dict:dict):
    """
    Generate random samples from property distributions.

    Parameters
    ----------
    material_property_dict : dict
        Dictionary mapping 'wet'/'dry' or 'both' to PropertyDistribution
        instances.

    Returns
    -------
    dict
        Sampled mean and variability values.
    """
    assert list(material_property_dict.keys()) == ['wet', 'dry'] or list(material_property_dict.keys()) == ['both'], "Material Property Dict must have either wet and dry properties or both property."
    sample_dict_prop_id = {}
    for key in material_property_dict.keys():
        mean = material_property_dict[key].mean_distribution.generate()
        if material_property_dict[key].stdev_distribution is not None:
            stdev = material_property_dict[key].stdev_distribution.generate()
        else:
            stdev = 0
        stdev_type = material_property_dict[key].stdev_type
        
        if material_property_dict[key].mean_slope_with_depth_distribution is not None:
            mean_slope_with_depth = material_property_dict[key].mean_slope_with_depth_distribution.generate()
        else:
            mean_slope_with_depth = 0
        
        sample_dict_prop_id[key] = {'mean': mean, 'mean_slope_with_depth': mean_slope_with_depth, 'stdev_or_cov': stdev, 'stdev_type':stdev_type}
        
    return sample_dict_prop_id