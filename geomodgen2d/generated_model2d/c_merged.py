"""
Module for Merged GeneratedModel2D
"""
from geomodgen2d.lithological_domain2d import LithologicalDomain2DCollection
from .a_each import GeneratedModel2D
from .b_collection import GeneratedProfileCollection2D
import warnings

class GeneratedModel2DMerged(GeneratedModel2D):
    """
    Represents a merged 2D generated model created from multiple profile sets
    in a `GeneratedProfileCollection2D`.
    """
    def __init__(self, generated_model_collection:GeneratedProfileCollection2D):
        """
        Initialize a merged 2D model from a profile collection.

        Parameters
        ----------
        generated_model_collection : GeneratedProfileCollection2D
            A collection of generated 2D profiles to merge.

        Notes
        -----
        This constructor computes:
        - The merged lithological domain
        - Merged simulated property profiles
        - Merged lithological ID → material mapping
        - Merged groundwater table depth
        """
        self.lit_domain, self.simulated_profiles, self.lit_id2material_dict, self.gwt_depth = self.__compute_merged_generated_profiles_fields(generated_model_collection)
        self.lit_order = -1
        self._locked = False #Same
        
    def __compute_merged_generated_profiles_fields(generated_model_collection:GeneratedProfileCollection2D):
        try:
            feature_ids = []
            for _,val in generated_model_collection._lit_id2material_dict.items():
                feature_id = str(val[0])
                if feature_id not in feature_ids:
                    feature_ids.append(feature_id)
        
            # Get ordered and lit_domain_set
            generated_model_collection.check()
            ordered_set_names_dict, lit_domain_set = generated_model_collection.get_ordered_set_names_dict()
            simulated_properties = generated_model_collection.get_simulated_properties
            
            merged_lit_domain, merged_all_lit_ids, gwt = LithologicalDomain2DCollection.get_merged_lit_domain(
                ordered=ordered_set_names_dict, lit_domain_set=lit_domain_set, valid_feature_ids=feature_ids
                )
            
            for key in simulated_properties:
                for order_id, set_name in ordered_set_names_dict.items():
                    simulated_profile = generated_model_collection.generated_model2d_set[set_name].simulated_profiles[key]
                    lit_domain = lit_domain_set[set_name]
                    
                    if order_id == 0:
                        simulated_profile_merged = None
                        
                    # Check ['X' and simulated_val_for_ignored_lit_property]
                    simulated_profile_merged = GeneratedProfileCollection2D.get_merged_simulated_profile(
                        order_id, current_merged_lit_domain=merged_lit_domain, current_merged_simulated_profile=simulated_profile_merged,
                        to_merge_lit_domain = lit_domain, to_merge_simulated_profile=simulated_profile, 
                        simulated_val_for_ignored_lit_property=generated_model_collection.generated_model2d_set[set_name].simulated_val_for_ignored_lit_property
                        )
            
            merged_lit_domain.lit_order = -1
            return merged_lit_domain, simulated_profile_merged, merged_all_lit_ids, gwt

        except Exception as e:
            warnings.warn(f"Failed to merge lithological domains: {e}")
            return None, None, None, None
    