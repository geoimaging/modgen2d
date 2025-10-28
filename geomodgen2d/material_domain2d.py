# This file is part of geomodgen3d a Python package for ...
# Copyright (C) XXXX Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
# LICENSE

"""Define a three-dimensional domain that defines a material."""

import numpy as np
import copy
from geomodgen2d.lithological_domain2d import LithologicalDomain2D
import geomodgen2d.general_functions as f
import geomodgen2d.random_generators as RandomGenerators
from geomodgen2d.material_properties import AllMainProperties

class MaterialDomain:
    """
    Class representing a 3D matrix (with layer_ID) with layers that can be created from a boundary or utility class.
    """
    def __init__(self, lithological_domain2D_instance_list: list, material_type_random_gen_dict: dict, all_main_properties_instance: AllMainProperties):
        """
        Initializes the MaterialDomain instance with the help of LithogolicalDomain3D instance and material_type_random_gen_dict.
        Make sure your lithological domain3D instance is finalized (with GWT defined.)
        
        Parameters:
        lithological_domain2D_instance list: List of lithological Domain Instances LithologicalDomain3D
            List of lithological domain3D instances. Note: when merged, the last one is the one with most priority (ie. replaced the former values)
                Properties imported: Domain3D (span_x, span_y, span_z, del_x, del_y, del_z), lithological_id, gwt  
                
        material_type_random_gen_dict: dict
            Material type random generator in dictionary format. Lithological_id will be randomized based on this dictionary.
            Format:
                {'def': RandomGenerator Instance, 
                 'U_': RandomGenerator Instance,
                 ...                 
                }        
        all_main_properties_instance (object): A AllMainProperties instance.
        """
        
        # 1) Check if material_type_random_gen_dict is in correct format.
        # a. All keys in correct format - CHECKED above
        # b. All values are random generator of type 'choice', and 'constant'.
        self.material_rnd_gen = material_type_random_gen_dict
        for key,val in material_type_random_gen_dict.items():
            assert isinstance(key, str), f"All keys in material_type_random_gen_dict must be a string. {key} is not a string."
            assert isinstance(val, RandomGenerators.RandomGenerator), 'All vals in material_type_random_gen_dict must be a RandomGenerator instance. The value for key {key} is not one.'
            assert val.rg_type in ['constant', 'choice'], f"The random generator must be defined with either constant or choice (discrete_choice, or discrete2continuous_pdf). Found {val.rg_type}"
        
        # 2) AllMainProperties and material_type_random_gen_dict are at sync.
        assert isinstance(all_main_properties_instance, AllMainProperties), "all_main_properties_instance must be a AllMainProperties instance"
        self.all_main_properties_instance = all_main_properties_instance
        
        all_main_properties_instance.check()  #Internal check
        
        self.features_id = material_type_random_gen_dict.keys()
        assert 'def' in self.features_id, "'def' must be in feature_id list (i.e., a key in material_type_random_gen_dict)."
        
        features_id_1 = self.features_id
        features_id_2 = all_main_properties_instance.features_w_id.keys()
        assert set(features_id_1).issubset(set(features_id_2)), f"features_id from AllMainProperties instance contains elements not present in material_type_random_gen_dict. Missing: {set(features_id_1) - set(features_id_2)}"

        # 3) Making sure the lithological_domain2D_instance_list is correct.        
        self.lithological_domain2D_instance_list = lithological_domain2D_instance_list
        
        ## Make sure it is a list with at least one instance
        if not lithological_domain2D_instance_list or not isinstance(lithological_domain2D_instance_list, list):
            raise ValueError("lithological_domain2D_instance_list must be a non-empty list.")

        # Check that all domain instances have the same span
        span_x, span_z = lithological_domain2D_instance_list[0].span_x, lithological_domain2D_instance_list[0].span_z
        for lit_domain in lithological_domain2D_instance_list:
            if (lit_domain.span_x, lit_domain.span_z) != (span_x, span_z):
                raise ValueError("All domain instances must have the same span_x, and span_z.")
        
        ## del_x,y,z of merged = smallest of the del_x,del_y,del_z
        # Compute the smallest resolution
        del_x = min([lit_domain.del_x for lit_domain in lithological_domain2D_instance_list])
        del_z = min([lit_domain.del_z for lit_domain in lithological_domain2D_instance_list])
        
        ## del_x/y/z / smallest must be an integer for all
        for lit_domain in lithological_domain2D_instance_list:
            if not (lit_domain.del_x / del_x).is_integer() or not (lit_domain.del_z / del_z).is_integer():
                raise ValueError("Each domain's del_x/z must be divisible by the smallest del_x/z.")

        ## gwt_depth of all instances must be same if not None.
        # Determine the common gwt_depth (or ensure consistency)
        gwt_depth_values = [lit_domain.gwt_depth for lit_domain in lithological_domain2D_instance_list]
        non_none_gwt_depths = [val for val in gwt_depth_values if val is not None]

        if non_none_gwt_depths:
            first_val = non_none_gwt_depths[0]
            if any(val != first_val for val in non_none_gwt_depths):
                raise ValueError("All non-None gwt_depth values must be the same.")
            self.gwt_depth = first_val
        else:
            self.gwt_depth = None
        
        ## Compute merged lithological domain (just for plotting and computing combined unique ids. During generation, provided lithological_domain2d_instance_list will be used.)
        ## Get lithological domain with minimum possible spacing (Will be used to get unique_ids)
        ref_instance = copy.deepcopy(lithological_domain2D_instance_list[0])
        ref_instance.remeshing_layered_matrix(del_x, del_z)
        for lit_domain in lithological_domain2D_instance_list[1:]:  
            lit_domain = copy.deepcopy(lit_domain)
            lit_domain.remeshing_layered_matrix(del_x, del_z)
            ref_instance.merge_with_another_lithological_domain(lit_domain)
            del lit_domain
        
        self.lithological_domain2D_combined_w_min_spacing = ref_instance
        self.span_x, self.span_z = span_x, span_z
        self.min_del_x, self.min_del_z = del_x, del_z
        
        # Seperate lithological_id as per features. i.e., 1,2,3 goes to 'def'; 'U_1, U_2' goes to 'U_'
        unique_values = np.unique(self.lithological_domain2D_combined_w_min_spacing.layered_matrix)
        lit_id = {feature_id: [] for feature_id in material_type_random_gen_dict.keys()}
        lit_id[9999] = []  # For unassigned values

        for val in unique_values:
            assigned = False
            for feature_id in self.features_id:
                if feature_id == 'def' and f.is_integer_value(val):
                    lit_id[feature_id].append(val)
                    assigned = True
                    break
                elif isinstance(val, str) and val.startswith(feature_id):
                    lit_id[feature_id].append(val)
                    assigned = True
                    break
            if not assigned:
                lit_id[9999].append(val)  # If never assigned   
                
        assert len(lit_id[9999]) == 0, f"The following unique values in lithological domain Not found in correct format, as per material_type_random_gen_dict keys(): {lit_id[9999]}"
        lit_id.pop(9999) # Remove this at the end.
        self.lithological_id = lit_id

        ## Finally, Assign each_lithological_id as per RandomGenerator
        id2material_dict = {}
        for key, val in lit_id.items():
            n = len(val)
            generated_types = material_type_random_gen_dict[key].generate((n,))
            for id_, mat_type in zip(val, generated_types):
                assert id_ not in id2material_dict.keys(), f'Duplicate ID Found. {id_} already assigned.'
                id2material_dict[id_] = [key, mat_type]
        
        self.lit_id2material_dict = id2material_dict
        self.sampled_properties = {}
        
    def get_sample_property(self, main_property_name):
        if main_property_name in self.sampled_properties.keys():
            raise AssertionError(f"Properties for main_property_name ({main_property_name}) already generated.")
        main_property_instance = self.all_main_properties_instance.main_properties[main_property_name]
        
        self.sampled_properties[main_property_name] = {}
        
        for key, material_metadata in self.lit_id2material_dict.items():
            feature_id, feature_material_name = material_metadata
            self.sampled_properties[main_property_name][key] = main_property_instance.generate_sample_dict(feature_id, feature_material_name, all_features_list = None) # TO DO all_features_list          
        
    def get_all_sample_properties(self, clear_saved = False):
        if clear_saved:
            self.sampled_properties = {}
            
        for main_property_name in self.all_main_properties_instance.main_properties.keys():
            self.get_sample_property(main_property_name)
        
    def plot():
        #Plot as per material type color
        #Show properties value if sampled.
        pass
    ## Breakdown into wet and dry properties based on GWT (will be done in spatial as per MainProperties Instance)
    ## Check material_types_choices during spatial simulation too.
    
