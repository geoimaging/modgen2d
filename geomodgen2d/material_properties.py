# Sanish Bhochhibhoya
# Apr 12: Changed the Structure of properties. 
#   Was: Main Property >> Feature ID >> wet/dry/both >> Dict of Random Generators or Random Generator
#   Now: Main Property >> Feature ID >> Dictionary of material types as keys >> Wet/dry/both >> Random Generator Instance.

import numpy as np
import geomodgen2d.random_generators as RandomGenerators
import geomodgen2d.general_functions as f
import pprint, re

class PropertyDistribution:
    """
        Manages material properties, ensuring validation and consistency.
    """
    def __init__(self, name, mean_distribution, stdev_distribution=None, stdev_type = 'stdev', description = ''):
        """
        Initializes the Properties object.

        Args:
            name (str): Name of the property.
            mean_distribution: RandomGenerator instance for mean. mean = a_m + b_m*z (but b_m is zero for current version)
            stdev_distribution: RandomGenerator instance for stdev (None if zero_variance).
            stdev_type (string, optional): Either 'stdev', or 'cov' if coefficient of variance. Note if cov is used, then stdev = a_m * cov

            description (string, optional): Any description. Can be any information about conditions for mean/stdev like "Soil Type", "Vs" (default: '').
        """
        self.name = name
        self.check_distribution(mean_distribution)
        self._mean_distribution = mean_distribution
        
        assert stdev_type in ['stdev', 'cov'], "stdev_type must be either 'stdev', or 'cov'."
        self.stdev_type = stdev_type
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
        
        if not isinstance(distribution_to_check, RandomGenerators.RandomGenerator):
            raise TypeError(f"set_dist must be a dictionary or an instance of a RandomGenerator class.")
            
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
    def mean_distribution(self):
        return self._mean_distribution

    @property
    def stdev_distribution(self):
        return self._stdev_distribution

    @property
    def description(self):
        return self._description

    @property
    def check(self):
        return self._check
        
class MaterialPropertySet:
    """
    Collection of materials and its wet and dry properties.
    """
    def __init__(self, main_property_name, feature_id):
        """
        Initializes the Material Property class for a material for one of the mainproperties.

        Args:
            main_property_name (str): Name of the main property
            feature_id (str): ID of the features of which the property is being defined.
    
        Hierarchy:
        Main_Properties_Set >> MainProperty instances >> Feature_ID >> material >> wet/dry_properties (Property instance.)
        
        """
        # These fields will be used for verification later.               
        self.main_property_name = main_property_name
        self.feature_id = feature_id
        self.material_property_dict = {}
        
    def add_material(self, material_name, property_distribution_instance, property_distribution_instance_if_dry=None):
        """
        Initializes the Material Property class for a material for one of the mainproperties.

        Args:
            name: 
            material_name (str): Name of the material
            property_distribution_instance (PropertyDistribution): Instance of the PropertyDistribution class to be added.
            property_distribution_instance (PropertyDistribution): Instance of the PropertyDistribution class to be added for its dry state (above GWT). If None, same property for both wet and dry.

        Hierarchy:
        AllMainProperties >> MainProperty instances >> Feature_ID >> MaterialPropertySet - materials >> wet/dry_properties (Property instance.)
        
        """
        assert material_name not in self.material_property_dict.keys(), f"Properties for {material_name} for property {self.main_property_name}, already defined."
        assert material_name !='layer0_val', "Material name cannot be layer0_val. Reserved Name"
        if not isinstance(property_distribution_instance, PropertyDistribution):
            raise TypeError("property_instance must be an instance of the PropertyDistribution class.")

        self.material_name = material_name
        property_distribution_instance.check_class()
        if property_distribution_instance_if_dry is not None:
            property_distribution_instance_if_dry.check_class()
            self.material_property_dict[material_name] = f.ReadOnlyDict({'wet': property_distribution_instance,
                                                                         'dry': property_distribution_instance_if_dry})
        else:
            self.material_property_dict[material_name] = f.ReadOnlyDict({'both': property_distribution_instance})     
            
    def check(self, material_property_dict = None):
        """
        material_property_dict: dictionary as same format as amterial_property_dict. This will overwrite the self.material_property_dict "only" for checking. Used in MainProperty Class.
        """
        if material_property_dict is not None:
            assert self.material_property_dict == {}, "MaterialPropertySet must have no material added for overwrite check feature."
        else:
            material_property_dict = self.material_property_dict
        
        material_names = list(material_property_dict.keys())
        if all(isinstance(cond, (int, float)) for cond in material_names):
            material_name_type = "numeric"
        elif all(isinstance(cond, str) for cond in material_names):
            material_name_type = "string"
        else:
            raise ValueError("material_names must be either all numbers or all strings.")
        
        # Check if material_names are unique
        if len(set(material_names)) != len(material_names):
            raise ValueError("material_names in material_property_instance_dict must be unique.")
        
        for material_name, material_property_dict in material_property_dict.items():
            if material_name == 'layer0_val': ## Used in overwriten material_property_dict
                assert isinstance(material_property_dict, (int, float)) or material_property_dict is None, f"layer0_val must be a number or None, but provided {material_property_dict}"
            else:
                assert list(material_property_dict.keys()) == ['wet', 'dry'] or list(material_property_dict.keys()) == ['both'], "Material Property must have either wet and dry properties or both property."
                for _, property_inst in material_property_dict.items():
                    property_inst.check_class()
        
class MainProperty:
    """
    Manages a main property, a property that is being generated.
    """
    def __init__(self, name, def_air_value=None, description='', features_w_id={'def':'soil'}):
        """
        Initializes the AllProperties class.

        Args:
            name: 
            description_list (string): Descriptions of main_property.
            def_air_value (int/floats, optional): Default property values for air(layer0). (Default: None; i.e. Error generated when this value is ever to be used.)
            features_w_id (dict, optional): Dictionary of features IDs (keys) and names (values).
        
        Hierarchy:
        Main_Properties_Set >> MainProperty instances >> Feature_ID >> material >> wet/dry_properties (Property instance.)
        
        """
        self.locked = False  #Locked to no edit (IE Checked.)
        
        validate_feature_keys(features_w_id)
        # Ensure name is string.
        if not isinstance(name, str):
            raise ValueError("main_properties_set must be a list of strings.")
        self.main_property_name = name

        # Ensure default air properties value is a number or None.
        if def_air_value is not None:
            assert isinstance(def_air_value, (int, float)), "def_air_value must have float values only, or is None (Air never to be generated in the model)"
        self.def_air_value = def_air_value
        
        if isinstance(features_w_id, dict):
            if len(features_w_id) == 0:
                raise ValueError("features_w_id cannot be empty dict")

            if 'def' not in features_w_id.keys():
                raise ValueError("'def' must be in features_w_id.keys")
            # Convert all keys and values to strings
            features_w_id = {str(k): str(v) for k, v in features_w_id.items()}
        else:
            raise ValueError("features_w_id must be a dictionary.")
        
        self.features_w_id = features_w_id
        # ALL VERIFIED. Now formatting all_properties instance accordingly.
        prop_dict = {'description':description}
        
        for id_,sub_d in features_w_id.items():
            if id_ == 'def':
                prop_dict[id_] = {'layer0_val':def_air_value} #Initialized as same property for wet and dry case (GWT)
            else:                
                prop_dict[id_] = {}

        self.main_property_dist = prop_dict

    def print(self):
        """Prints the main properties, subdivisions with IDs, and all stored properties."""
        print(f"Property Name: {self.main_property_name}")
        # print(f"Features_w_id: {self.features_w_id}")
        # print("Properties:")
        pprint.pp(self.main_property_dist)
    
    def add_material_property_set(self, feature_id, material_property_set_instance):
        """
        Adds a property instance to the specified subdivision and main property category. 
        
        Parameters:
        - feature_id (str): ID of the features of which the property is being defined.
        - material_property_set_instance (MaterialPropertySet): MaterialPropertySet Instance,
    
        Raises:
        - TypeError: If property_distribution_instance is not an instance of PropertyDistribution.
        - ValueError: If feature_id is not found in features_w_id.
        """
        # main_property_name must be main_properties_set. and sub_division_id must be in self.sub_division_w_id
        assert not self.locked, "The MainProperty instance is locked. Overwrite it to unlock and add."
        
        if feature_id not in self.features_w_id.keys():
            raise f"ERROR: feature_id ({feature_id}) is not in sub_divisions_w_id"
        
        # Check if set_dist is dictionary or not
        # assert isinstance(material_property_set_instance, MaterialPropertySet), "material_property_set_instance must be a MaterialPropertySet Instance."
        material_property_set_instance.check()
        
        assert material_property_set_instance.feature_id == feature_id, f"feature_id provided, {feature_id} does not match with that in material_property instance {material_property_set_instance.feature_id}."
        
        addn_keys = set(self.main_property_dist[feature_id].keys())-set(['layer0_val'])
        
        assert len(addn_keys) == 0, f"material_property_set already added for feature_id = {feature_id} of main_property {self.main_property_name}"
        for material_name, prop_dict in material_property_set_instance.material_property_dict.items():
            self.main_property_dist[feature_id][material_name] = prop_dict
                
    def __getattr__(self, name):
        """
        Enables attribute access to property groups.
    
        Example:
        >>> obj.vs_utils.mean_distribution
        Will return obj.properties["vs_utils"]["mean_distribution"] if it exists.
    
        Raises:
        - AttributeError: If the requested property group is not found.
        """
            
        # Allows attribute access like a.vs_utils.mean_distribution
        if name in self.main_property_dist:
            return self.main_property_dist[name]
        raise AttributeError(f"'MainProperty object has no attribute '{name}'")

    def lock_and_check(self):
        """
        Performs validation checks on the PropertyDistributions of the main property.
    
        Parameters:
        - detailed_check (bool): If True, performs a more in-depth check using `check_class()`.
    
        Raises:
        - TypeError: If a property instance is not of type PropertyDistribution.
        - AssertionError: If `check` is not performed on a PropertyDistribution instance.
        - AssertionError: If 'layer0_val' is not a number or None.
        """
        validate_feature_keys(self.features_w_id)
        for id_ in self.main_property_dist.keys():
            if id_ not in ['description']:
                material_property_instance = MaterialPropertySet(self.main_property_name, id_)
                material_property_instance.check(self.main_property_dist[id_])
                
        self.locked = True

    #@property
    def generate_sample_dict(self, feature_id, feature_material_name, all_features_list = None):
        """
        Generates a dictionary containing sampled property values using provided distributions in allproperty class.
    
        Ensures that all property instances have passed validation before generating the dictionary.
    
        Returns:
        - dict: A read-only dictionary containing sampled property data.
    
        Raises:
        - TypeError: If a property instance is not of type Properties.
        - AssertionError: If 'layer0_val' is not a number or None.
        """
        assert self.locked, "Cannot generate dictionary until the MainProperty instance is locked (checked)"
        self.lock_and_check() 
        
        assert feature_id not in ['description', ''], "Invalid feature id"
        assert feature_id in self.main_property_dist.keys(), f'Provided feature id {feature_id} not in properties.keys()'
        
        features_in_dict = list(self.main_property_dist[feature_id].keys())
        assert feature_material_name in features_in_dict, f"feature_id ({feature_material_name}) not in main_property_distribution's features. features available: {features_in_dict}"
        
        if all_features_list is not None:
            missing_features = set(all_features_list) - set(features_in_dict)
            assert set(all_features_list).issubset(set(features_in_dict)), f"Missing elements: {missing_features}"
            
        sample_dict_prop_id = get_random_generated_sample(self.main_property_dist[feature_id][feature_material_name])
         
        if 'layer0_val' in self.main_property_dist[feature_id].keys():
            sample_dict_prop_id['layer0_air'] = {'mean':self.main_property_dist[feature_id]['layer0_val']}
            
        return f.ReadOnlyDict(f.ReadOnlyDict(sample_dict_prop_id))
    
class AllMainProperties:
    """
    Collection of all main properties, and their features' probability distributions.
    """
    def __init__(self, features_w_id = {'def':'soil'}):
        """
        Initializes the AllMainProperties class.

        Args:
            features_w_id (dict, optional): Dictionary of features IDs (keys) and names (values).

        Format:
        Each mainproperty is a key, and values are MainProperty Instance.

        Hierarchy:
        AllMainProperties >> MainProperty Instance ( >Sub_divisions >> Property instance)
        """
        self.features_w_id = features_w_id
        validate_feature_keys(features_w_id)
        self.main_properties = {}
        
    def add_main_property(self, main_property_name, material_property_set_instance_in_list, description = '', def_air_value=0):
        assert main_property_name not in self.main_properties.keys(), f"'{main_property_name}' already exists in the AllMainProperties Instance"
        self.main_properties[main_property_name] = MainProperty(main_property_name, def_air_value, description, self.features_w_id)
        assert isinstance(material_property_set_instance_in_list, list), "material_properties_instances_list must be a list"
        for mat_prop_set_instance in material_property_set_instance_in_list:
            # assert type(mat_prop_set_instance) is MaterialPropertySet, f"mat_prop_set_instance must be a MaterialPropertySet Instance. Provided {type(mat_prop_set_instance)}"
            assert mat_prop_set_instance.main_property_name == main_property_name, f"Main property name on MaterialProperty instance ({mat_prop_set_instance.main_property_name}) does not match with the name provided {main_property_name}"
            self.main_properties[main_property_name].add_material_property_set(mat_prop_set_instance.feature_id, mat_prop_set_instance)
        self.main_properties[main_property_name].lock_and_check()
        
    def print(self):
        """Prints the main properties, subdivisions with IDs, and all stored properties."""
        print(f"Main_Properties: {self.main_properties.keys()}")
        print(f"features_w_id: {self.features_w_id}")
        for key in self.main_properties.keys():
            print("---------------------------")
            self.main_properties[key].print()

    def check(self):
        """
        Performs validation checks on all stored main properties.
    
        Raises:
        - TypeError: If a property instance is not of type Properties.
        - AssertionError: If `check` is not performed on a property instance.
        - AssertionError: If 'layer0_val' is not a number or None.
        """
        for main_property_name in self.main_properties.keys():
            self.main_properties[main_property_name].lock_and_check()
            
class AllAdditionalProperties:
    """
    Collection of all additional properties, not generated properties but required for model generation like number of layers, etc. (Optional)
    """
    def __init__(self):
        """
        Initializes the AllAdditionalProperties class. Note, here the values are RandomGeneratorInstance mean rather than PropertyDistribution instance.
        """
        self.addn_properties = {}
        
    def add_addn_property(self, name, randomGeneratorInstance_mean, description=''):
        assert name not in self.addn_properties.keys(), f"'{name}' already exists in the AllAdditionalProperties Instance"
        self.addn_properties[name] = randomGeneratorInstance_mean
            
    def print(self):
        """Prints the additional properties, subdivisions with IDs, and all stored properties."""
        print(f"Additional_Properties: {self.addn_properties.keys()}")
        
        
def get_random_generated_sample(material_property_dict):
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

def validate_feature_keys(features_w_id):
    keys = list(features_w_id.keys())

    # Rule 1: key must end with underscore "_"
    # Rule 2: Key must not contain digits,
    # Rule 3: Key must not contain other underscores.
    # Rule 4: One of the key must be exactly "def"
    invalid = [
        k for k in keys
        if k != 'def' and (
            not k.endswith('_') or
            re.search(r'\d', k) or
            k[:-1].count('_') > 0  # only one underscore allowed at end
        )
    ]
    if invalid:
        raise ValueError(f"Invalid keys (must end with '_', no digits, no other underscores): {invalid}")
        
    if 'def' not in keys:
        raise ValueError("Missing required key: 'def'")
    
    if '' in keys:
        raise ValueError("Key cannot be ''")
    
    # assert feature_id not in ['description', ''], 'Invalid feature_id.'  Since both do not end with "_", should not be the case but.
    
