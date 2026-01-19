import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
# from IPython.display import clear_output
import numpy as np
import geomodgen2d.general_functions as f
from geomodgen2d.lithological_domain2d import LithologicalDomain2D, LithologicalDomain2DFromObstruction2D, LithologicalDomain2DReadOnly, LithologicalDomain2DCollection
from .a_each import GeneratedModel2D
from .b_collection import GeneratedProfileCollection2D
import warnings

class GeneratedModel2DMerged(GeneratedModel2D):
    def __init__(self, generated_model_collection:GeneratedProfileCollection2D):
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
            ordered = {}
            lit_domain_set = {}
            simulated_properties = None
            
            for set_name, gen_model2d in generated_model_collection._generated_model2d_set.items():
                lit_domain = gen_model2d.lit_domain
                lit_order = gen_model2d.lit_order
                
                # Enforce uniqueness
                if lit_order in ordered:
                    raise ValueError(f"Duplicate lit_order detected: {lit_order}")
                if set_name in ordered.values():
                    raise ValueError(f"Duplicate set_name detected: {set_name}")
                
                ordered[lit_order] = set_name
                lit_domain_set[set_name] = lit_domain
                
                simulated_properties_each = set(generated_model_collection.generated_model2d_set[set_name].simulated_profiles.keys())
                if simulated_properties is None:
                    simulated_properties = simulated_properties_each
                    
                if simulated_properties_each != simulated_properties:
                    raise ValueError(
                        f"Simulated properties mismatch for set '{set_name}'. "
                        f"Expected {sorted(simulated_properties)}, "
                        f"found {sorted(simulated_properties_each)}."
                    )
                    
            # Stable sort by lit_order (ascending)
            ordered = sorted(ordered.items(), key=lambda x: x[0])

            merged_lit_domain, merged_all_lit_ids, gwt = LithologicalDomain2DCollection.get_merged_lit_domain(
                ordered=ordered, lit_domain_set=lit_domain_set, valid_feature_ids=feature_ids
                )
            
            for key in simulated_properties:
                for order_id, set_name in ordered.items():
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
    