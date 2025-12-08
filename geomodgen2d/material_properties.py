# Sanish Bhochhibhoya
# Apr 12: Changed the Structure of properties. 
#   Was: Main Property >> Feature ID >> wet/dry/both >> Dict of Random Generators or Random Generator
#   Now: Main Property >> Feature ID >> Dictionary of material types as keys >> Wet/dry/both >> Random Generator Instance.

from geomodgen2d.features_config import FeaturesConfig
from geomodgen2d.material_property_each import MainProperty

class MainPropertiesConfig:
    """
    Collection of all main properties, and their features' probability distributions.
    """
    def __init__(self, features_config:FeaturesConfig, layer0_flag:bool=False):
        """
        Initializes the MainPropertiesConfig class.

        Args:
            features_config: FeaturesConfig instance
            A config describing all features, their ids, and material_type random generator.
                    
        Hierarchy:
        MainPropertiesConfig >> MainProperty Instance ( >Sub_divisions >> Property instance)
        """
        
        if not isinstance(features_config, FeaturesConfig):
            raise TypeError("features_config must be a FeaturesConfig Instance.")
        
        self._features_config = features_config
        self._layer0_flag = layer0_flag
        self._main_properties = []
    
    @property
    def features_config(self):
        return self._features_config
    
    @property
    def layer0_flag(self):
        return self._layer0_flag
        
    @property
    def main_properties(self):
        return self._main_properties
    
    @property
    def get_main_properties_names(self):
        list_ = []
        for mp in self.main_properties:
            list_.append(self.main_properties.main_property_name)
        return list_
    
    def add_main_property(self, main_property_instance:MainProperty):
        main_property_name = main_property_instance.main_property_name
        if main_property_name not in self.get_main_properties_names:
            raise ValueError(f"'{main_property_instance.main_property_name}' already exists in the MainPropertiesConfig Instance")
        main_property_instance.lock_and_check()
        self._main_properties.append(main_property_instance)
        
    def print(self):
        """Prints the main properties, subdivisions with IDs, and all stored properties."""
        print(f"Main_Properties: {self.get_main_properties_names}")
        print(f"feature_ids: {self.features_config.get_feature_ids()}")
        for mp in self.main_properties:
            print("---------------------------")
            mp.print()

    def check(self):
        """
        Performs validation checks on all stored main properties.
    
        Raises:
        - TypeError: If a property instance is not of type Properties.
        - AssertionError: If `check` is not performed on a property instance.
        - AssertionError: If 'layer0_val' is not a number or None.
        """
        for main_property in self.main_properties:
            main_property.lock_and_check()
            
class AuxillaryProperties:
    """
    Collection of all auxillary properties, not generated properties but required for model generation like number of layers, etc. (Optional)
    """
    def __init__(self):
        """
        Initializes the AuxillaryProperties class. Note, here the values are RandomGeneratorInstance mean rather than PropertyDistribution instance.
        """
        self.aux_properties = {}
        
    def add_aux_property(self, name, randomGeneratorInstance_mean):
        assert name not in self.aux_properties.keys(), f"'{name}' already exists in the AuxillaryProperties Instance"
        self.aux_properties[name] = randomGeneratorInstance_mean
            
    def print(self):
        print(f"Additional_Properties: {self.aux_properties.keys()}")
        

    
