# Sanish Bhochhibhoya
# Apr 12: Changed the Structure of properties. 
#   Was: Main Property >> Feature ID >> wet/dry/both >> Dict of Random Generators or Random Generator
#   Now: Main Property >> Feature ID >> Dictionary of material types as keys >> Wet/dry/both >> Random Generator Instance.

from geomodgen2d.features_config import FeaturesConfig
from geomodgen2d.main_property_each import MainProperty
from geomodgen2d.lithological_domain2d import LithologicalDomain2DCollection
import geomodgen2d.general_functions as f
import numpy as np

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
        self._main_properties = {}
        
        self._locked = False
        self._lit_id2material_dict = {}
        self._sampled_properties = {}
        self._lit_collection_unique_code = 0
        self._unique_code = 0
    
    @property
    def features_config(self):
        return self._features_config
    
    def get_feature_ids(self):
        return self.features_config.get_feature_ids()
    
    @property
    def layer0_flag(self):
        return self._layer0_flag
        
    @property
    def main_properties(self):
        return self._main_properties
    
    def get_main_properties_names(self):
        return list(self.main_properties.keys())
    
    @property
    def lit_id2material_dict(self):
        return self._lit_id2material_dict
    
    @property
    def sampled_properties(self):
        return self._sampled_properties
    
    @property
    def unique_code(self):
        return self._unique_code
    
    def unlock(self):
        self._locked = False
        self._lit_id2material_dict = {}
        self._sampled_properties = {}
        self._unique_code = 0
        self._lit_collection_unique_code = 0
        
    def add_main_property(self, main_property_instance:MainProperty):
        main_property_name = main_property_instance.main_property_name
        if main_property_name in self.get_main_properties_names():
            raise ValueError(f"'{main_property_name}' already exists in the MainPropertiesConfig Instance")
        main_property_instance.lock_and_check()
        
        # Make sure features_config and layer0 are same at each main_property_instance.
        
        self._main_properties[main_property_name] = main_property_instance
        
    def print(self):
        """Prints the main properties, subdivisions with IDs, and all stored properties."""
        print(f"Main_Properties: {self.get_main_properties_names()}")
        print(f"feature_ids: {self.features_config.get_feature_ids()}")
        for _,mp in self.main_properties.items():
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
            
        names = self.get_main_properties_names()
        # make sure no name in get_names is duplicate, or "", or non-string
        for name in names:
            if not isinstance(name, str):
                raise TypeError(f"Main property name must be a string, got {type(name)}")

            if name.strip() == "":
                raise ValueError("Main property name cannot be empty or whitespace")

        if len(names) != len(set(names)):
            raise ValueError("Duplicate main property names detected")
        
        for _,main_property in self.main_properties.items():
            main_property.lock_and_check()
            
    def lock_and_generate_sample_properties(self, lit_domain2d_collection_instance:LithologicalDomain2DCollection):
        if self._locked:
            raise ValueError("Sample properties already generated. Try .unlock to generate again.")
        
        self.unlock()
        self.check()
        if not isinstance(lit_domain2d_collection_instance, LithologicalDomain2DCollection):
            raise TypeError("lit_domain2d_collection_instance must be LithologicalDomain2dCollection instance.")
        
        if not lit_domain2d_collection_instance._locked:
            raise ValueError("lit_domain2d_collection_instance is not locked yet. First lock first using .lock.")
        
        ## Get id2material_dict.
        # Lock GenerateProfile2D. and add litid2materialproperties.
        id2material_dict = {}
        for feature_id, vals in lit_domain2d_collection_instance.all_lit_ids.items():
            material_type_random_gen_dict = self.features_config.get_material_types_distribution(feature_id)
            n = len(vals)
            generated_types = material_type_random_gen_dict.generate((n,))
            code = f.format_lith_ids(feature_id, vals)
            for id_, mat_type in zip(code, generated_types):
                if id_  in id2material_dict.keys(): 
                    raise RuntimeError(f'Duplicate lit_ID Found. {id_} already assigned.')
                
                if id_ == '0':
                    mat_type = 'layer0'
                    
                id2material_dict[id_] = np.array([feature_id, mat_type])
                
        self._lit_id2material_dict = id2material_dict
        # for set_name in self.generated_model_set.keys():
        #     generated_profile_instance = self.generated_model_set[set_name]
        #     if not isinstance(generated_profile_instance, GeneratedModel2D):
        #         raise TypeError(f"The values of all generated_profiles_set must be GeneratedModel2D. Not the case for {set_name}")
        #     generated_profile_instance.lock_and_set_lit_id2material(self.lit_id2material_dict)           
        # self._lit_id2material_dict = id2material_dict
        
        for main_property_name in self.main_properties.keys():
            self._get_sample_property(main_property_name)
        
        self._lit_collection_unique_code = lit_domain2d_collection_instance.unique_code
        self._locked = True
        self._unique_code = np.random.randint(1, 10**18, dtype=np.int64)
        #self.check()
        
    def _get_sample_property(self, main_property_name):
        if main_property_name in self.sampled_properties.keys():
            raise AssertionError(f"Properties for main_property_name ({main_property_name}) already generated.")
        main_property_instance = self.main_properties[main_property_name]
        
        self._sampled_properties[main_property_name] = {}
        for key, material_metadata in self.lit_id2material_dict.items():
            feature_id, feature_material_name = material_metadata
            self._sampled_properties[main_property_name][key] = main_property_instance.generate_sample_dict(feature_id, feature_material_name)
            ##TODO
    
    def check_consistent_with_lit_domain2d_collection(self, lit_domain2d_collection_instance:LithologicalDomain2DCollection):
        check_consistent_with_lit_domain2d_collection(lit_domain2d_collection_instance, self.lit_id2material_dict, self.sampled_properties, self.get_main_properties_names())
        if self._lit_collection_unique_code != lit_domain2d_collection_instance.unique_code:
            raise ValueError("While lit_domain2d is consistent, unique code mismatch. It means the sampled properties has been changed. Generate sample properties again for consistency.")
        
def check_consistent_with_lit_domain2d_collection(lit_domain2d_collection_instance:LithologicalDomain2DCollection, lit_id2material_dict:dict, sampled_properties:dict, main_property_list:list):
    if not isinstance(lit_domain2d_collection_instance, LithologicalDomain2DCollection):
        raise TypeError("lit_domain2d_collection_instance must be LithologicalDomain2dCollection instance.")
    
    if not lit_domain2d_collection_instance._locked:
        raise ValueError("lit_domain2d_collection_instance is not locked yet. First lock first using .lock.")
    
    all_main_properties = set(sampled_properties.keys())
    expected_main_properties = set(main_property_list)

    if all_main_properties != expected_main_properties:
        raise ValueError(
            "Mismatch between sampled_properties keys and main_property_list.\n"
            f"Missing: {expected_main_properties - all_main_properties}\n"
            f"Extra: {all_main_properties - expected_main_properties}"
        )

    # --- Layer / lithology ID consistency ---
    expected_layer_ids = set(lit_id2material_dict.keys())

    for prop_name, each_sample_property in sampled_properties.items():
        # Validate structure of each processed property
        f.validate_processed_property_dict(each_sample_property)

        layer_ids = set(each_sample_property.keys())

        if layer_ids != expected_layer_ids:
            raise ValueError(
                f"Layer ID mismatch for property '{prop_name}'.\n"
                f"Missing: {expected_layer_ids - layer_ids}\n"
                f"Extra: {layer_ids - expected_layer_ids}"
            )

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
        

    
