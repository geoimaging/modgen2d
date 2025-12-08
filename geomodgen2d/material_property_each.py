# Sanish Bhochhibhoya
# Apr 12: Changed the Structure of properties. 
#   Was: Main Property >> Feature ID >> wet/dry/both >> Dict of Random Generators or Random Generator
#   Now: Main Property >> Feature ID >> Dictionary of material types as keys >> Wet/dry/both >> Random Generator Instance.

from geomodgen2d.random_generators import RandomGeneratorAbstract
from geomodgen2d.features_config import FeaturesConfig
import geomodgen2d.general_functions as f
import pprint

class PropertyDistribution:
    __slots__ = ['_property_name', '_mean_distribution', '_stdev_distribution', '_stdev_type', '_description', '_check']
    """
        Manages material properties, ensuring validation and consistency.
    """
    def __init__(self, property_name, mean_distribution, stdev_distribution=None, stdev_type = 'stdev', description = ''):
        """
        Initializes the Properties object.

        Args:
            property_name (str): Name of the property.
            mean_distribution: RandomGenerator instance for mean. mean = a_m + b_m*z (but b_m is zero for current version)
            stdev_distribution: RandomGenerator instance for stdev (None if zero_variance).
            stdev_type (string, optional): Either 'stdev', or 'cov' if coefficient of variance. Note if cov is used, then stdev = a_m * cov

            description (string, optional): Any description. Can be any information about conditions for mean/stdev like "Soil Type", "Vs" (default: '').
        """
        self._property_name = property_name
        self.check_distribution(mean_distribution)
        self._mean_distribution = mean_distribution
        
        if stdev_type not in ['stdev', 'cov']:
            raise ValueError("stdev_type must be either 'stdev', or 'cov'.")
        
        self._stdev_type = stdev_type
        if stdev_distribution is None:
            self._stdev_distribution = None 
        else:
            self.check_distribution(stdev_distribution)
            self._stdev_distribution = stdev_distribution 

        self._description = description # Condition details: Eg. If mean or variance is dependent to anything, if so will have a user-understandable conditions mentioned like "soil type", "Vs > "
        self._check = False # Validation check flag

    def check_distribution(self, distribution_to_check):
        """
        Validates if the given distribution is correctly formatted.

        Args:
            set_dist: RandomGenerator instance to be set.
            curr_dist: Existing distribution (self.mean_distribution/self.stdev_distribution), if any (default: None). Generally used to make sure curr_dist is "NA" before setting its value to set_dist. 

        """
        
        if not isinstance(distribution_to_check, RandomGeneratorAbstract):
            raise TypeError(f"set_dist must be an instance of a subclass of a RandomGenerator class.")
            
    def check_class(self):
        """
        Performs internal validation on the distribution assignments.

        Checks:
        1) Mean distribution must not be 'NA'; like when initialized.
        2) Both Std and cov distribution must not be 'NA'; but both cannot be defined too. 
        3) The distributions (mean, or one of std/cov) is correctly defined.
        """
        self._check = False
        
        self.check_distribution(self._mean_distribution)
        if self._stdev_distribution is not None:
            self.check_distribution(self._stdev_distribution)
        self._check = True
            
    @property
    def property_name(self):
        return self._property_name
    
    @property
    def mean_distribution(self):
        return self._mean_distribution

    @property
    def stdev_distribution(self):
        return self._stdev_distribution

    @property
    def stdev_type(self):
        return self._stdev_type

    @property
    def description(self):
        return self._description

    @property
    def check(self):
        return self._check
        
class MainProperty:
    __slots__ = ['_locked', '_main_property_name', '_features_config', '_layer0_flag', '_main_property_dist', '_description']
    """
    Manages a main property, a property that is being generated.
    """
    def __init__(self, main_property_name, features_config:FeaturesConfig, layer0_flag:bool=False, description=''):
        """
        Initializes the MainProperty class. 
        
        Note: Must have a layer0 (int/floats, optional): Default property values for air/water(layer0). (Default: None; i.e. Error generated when this value is ever to be used.)

        Args:
            name: 
            description (string): Descriptions of main_property.
            all_feature_ids_list (list, optional): List of all feature IDs.
        
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
        Initializes the Material Property class for a material for one of the mainproperties.

        Args:
            feature_id (str): ID of the feature of which the property is being defined.
            material_name (str): Name of the material
            
            property_distribution_instance (PropertyDistribution): Instance of the PropertyDistribution class to be added.
            property_distribution_instance_if_dry (PropertyDistribution): Instance of the PropertyDistribution class to be added for its dry state (above GWT). If None, same property for both wet and dry.

        Hierarchy:
        AllMainProperties >> MainProperty instances >> Feature_ID >> MaterialPropertySet - materials >> wet/dry_properties (Property instance.)
        
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
        Performs validation checks on the PropertyDistributions of the main property.
    
        Parameters:
        - detailed_check (bool): If True, performs a more in-depth check using `check_class()`.
    
        Raises:
        - TypeError: If a property instance is not of type PropertyDistribution.
        - AssertionError: If `check` is not performed on a PropertyDistribution instance.
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
    def generate_sample_dict(self, feature_id, feature_material_name, all_feature_ids_list = None):
        """
        Generates a dictionary containing sampled property values using provided distributions in allproperty class.
    
        Ensures that all property instances have passed validation before generating the dictionary.
    
        Returns:
        - dict: A read-only dictionary containing sampled property data.
    
        Raises:
        - TypeError: If a property instance is not of type Properties.
        - AssertionError: If 'layer0' is not a number or None.
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
        
        if all_feature_ids_list is not None:
            missing_features = set(all_feature_ids_list) - set(features_in_dict)
            if not set(all_feature_ids_list).issubset(set(features_in_dict)):
                raise ValueError(f"Missing elements: {missing_features}")
            
        sample_dict_prop_id = get_random_generated_sample(self.main_property_dist[feature_id][feature_material_name])
         
        return f.FixedKeyDict(sample_dict_prop_id)
    
def get_random_generated_sample(material_property_dict:dict):
    """
    Retrieves a random generated sample based on the given probability distribution.

    Parameters:
        material_property_dict (dict): 
            A dictionary mapping wet/dry, or both to PropertyDistribution instances. 

    Returns:
        Dictionary: A randomly generated samples (mean, and std.) based on the respective prob. distribution in dictionary format
    """
    assert list(material_property_dict.keys()) == ['wet', 'dry'] or list(material_property_dict.keys()) == ['both'], "Material Property Dict must have either wet and dry properties or both property."
    sample_dict_prop_id = {}
    for key in material_property_dict.keys():
        mean = material_property_dict[key].mean_distribution.generate()[0]
        if material_property_dict[key].stdev_distribution is not None:
            stdev = material_property_dict[key].stdev_distribution.generate()[0]
        else:
            stdev = 0
        stdev_type = material_property_dict[key].stdev_type
        
        sample_dict_prop_id[key] = {'mean': mean, 'mean_bm': 0, 'stdev/cov': stdev, 'stdev_type':stdev_type}
        
    return sample_dict_prop_id