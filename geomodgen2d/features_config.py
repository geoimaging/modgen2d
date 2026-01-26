# This file is part of geomodgen3d a Python package for ...
# Copyright (C) XXXX Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
# LICENSE

"""Feature configuration utilities."""

import numpy as np
from geomodgen2d.general_functions import validate_feature_ids_list, is_valid_feature_id
from geomodgen2d.random_generators import RandomGeneratorAbstract, DiscreteChoice, Constant

class FeaturesConfig:
    """
    Container for feature identifiers and their material distributions. Stores feature_id, 
    feature description, and RandomGenerator for each feature.
    """
    __slots__ = ['_feature_ids_mapping'] 

    def __init__(self):
        """Initialize an empty 'FeaturesConfig' class."""
        self._feature_ids_mapping = {}
        
    def reset(self):
        """Remove all stored features."""
        self._feature_ids_mapping = {}
    
    def add_feature(self, feature_id:str, 
                    material_type_distribution:RandomGeneratorAbstract,
                    feature_description:str = None
                    ):
        """
        Add a feature definition.

        Parameters
        ----------
        feature_id : str
            Feature identifier.
        material_type_distribution : RandomGeneratorAbstract
            Distribution defining material types.
        feature_description : str, optional
            Human-readable feature description.

        Raises
        ------
        KeyError
            If feature_id is invalid or already exists.
        TypeError
            If material_type_distribution has an invalid type.
        ValueError
            If material types are invalid.
        """
        
        valid_prefix, msg = is_valid_feature_id(feature_id)
        if not valid_prefix and feature_id != 'def':
            raise KeyError(msg)
        
        if feature_id in self._feature_ids_mapping.keys():
            raise KeyError(f"{feature_id} already added. Available feature_ids: {self.get_feature_ids()}. Remove it before if needed.")
        
        feature_description = "" if feature_description is None else feature_description    
        if not isinstance(feature_description, str):
            raise ValueError (f"All feature_description must be a string. {feature_description} is not a string.")
        
        if not isinstance(material_type_distribution, Constant) and not isinstance(material_type_distribution, DiscreteChoice):
            raise TypeError('A random generator defining material properties in feature_ids_materialtype_mapping must be a either DiscreteChoice or Constant instances.')
        
        if isinstance(material_type_distribution, Constant):
            props = [material_type_distribution.value]
        elif isinstance(material_type_distribution, DiscreteChoice):
            props = list(material_type_distribution.x)
        
        # must all strings
        if not (
            all(isinstance(x, str) for x in props)
        ):
            raise ValueError(f"material_names must be either all strings. Not that case in for feature_id: '{feature_id}'")

        if 'layer0' in props:
            raise KeyError("Cannot have 'layer0' in distribution. Note: Define layer0's distribution directly in MainProperties Class.")

        # --- Assert NO duplicates in props ---
        if len(props) != len(set(props)):
            raise ValueError(
                f"Duplicate values found in material_type_distribution for feature '{feature_id}'. "
                f"Duplicates: { [p for p in props if props.count(p) > 1] }"
            )
        
        clean_props = [
            str(p) if isinstance(p, (np.str_, np.bytes_)) else  # convert NumPy strings → Python str
            p.item() if isinstance(p, np.generic) else           # convert other NumPy scalars → native Python numbers
            p                                                  # leave everything else as-is
            for p in props
        ]
        
        self._feature_ids_mapping[feature_id] = {
            'description':feature_description,
            'material_type_distribution': material_type_distribution,
            'material_type_list': clean_props,
        }
        
    def remove_feature(self, feature_id:str):
        """
        Remove a feature.

        Parameters
        ----------
        feature_id : str
            Feature identifier.

        Raises
        ------
        KeyError
            If feature_id does not exist.
        """
        if feature_id not in self._feature_ids_mapping.keys():
            raise KeyError(f"{feature_id} not added yet, so failed removing. Available feature_ids: {self.get_feature_ids()}.")
        self._feature_ids_mapping.pop(feature_id)
               
    def check(self):
        """
        Validate internal feature configuration.

        Raises
        ------
        ValueError
            If feature definitions are inconsistent.
        TypeError
            If invalid gene
        """
        # 1) Check if _feature_ids_mapping is in correct format.
        # a. All keys in correct format 
        # b. All generators are random generator of type 'Constant', and 'DiscreteChoice'.
        # c. All material list (xs in random generator) are all strings
        # d. material list are consistent
        
        validate_feature_ids_list(self.get_feature_ids())
        for feature_id, val in self._feature_ids_mapping.items():    
            if not isinstance(val, dict):
                raise SystemError("Invalid format of FeaturesConfig instance")
            
            gen = val['material_type_distribution']
            if not isinstance(gen, (Constant, DiscreteChoice)):
                raise TypeError(f'A random generator defining material properties in FeaturesConfig must be a either DiscreteChoice or Constant instances. The value for feature_id {feature_id} is not one.')
        
            if not isinstance(val['description'], str):
                raise ValueError (f"All descriptions in FeaturesConfig must be a string. {val['description']} is not a string.")
            
            if isinstance(gen, Constant):
                props = [gen.value]
            elif isinstance(gen, DiscreteChoice):
                props = list(gen.x)
                
            # make sure both list props and val['material_type_list'] are same
            stored_list = val['material_type_list']
            
            # Check all elements have same type and have same values
            if len(set(type(x) for x in props)) != 1 or len(set(type(x) for x in stored_list)) != 1:
                raise TypeError("Elements in props or stored_list are not all the same type")

            if set(props) != set(stored_list):
                raise ValueError(
                    f"Properties mismatch for feature_id: '{feature_id}'. "
                    f"Generator properties: {set(props)} vs stored material types: {set(stored_list)}"
                )
                
            # must all strings
            if not (
                all(isinstance(x, str) for x in stored_list)
            ):
                raise ValueError(f"material_names must be either all strings. Not that case in for feature_id: '{feature_id}'")

            # Check if material_names are unique
            if len(set(stored_list)) != len(stored_list):
                raise ValueError("Material_names in material_property_instance_dict must be unique.")
            
            if 'layer0' in props:
                raise ValueError(f"Cannot have 'layer0' in distribution. Found in feature_id:'{feature_id}' Note: Define layer0's distribution directly in MainProperties Class.")
                
    def get_feature_ids(self):
        """
        Return all feature identifiers.

        Returns
        -------
        list of str
            Feature IDs.
        """
        return list(self._feature_ids_mapping.keys())
    
    def get_feature_descriptions(self):
        """
        Return feature descriptions.

        Returns
        -------
        dict
            Mapping of feature_id to description.
        """
        dict_ = {}
        for key, val in self._feature_ids_mapping.items():
            dict_[key] = val['description']
        return dict_
    
    def get_material_types_distribution(self, feature_id):
        """
        Return material type distribution for a feature.

        Parameters
        ----------
        feature_id : str
            Feature identifier.

        Returns
        -------
        RandomGeneratorAbstract
            Material type distribution.

        Raises
        ------
        KeyError
            If feature_id does not exist.
        """
        self.check()
        if feature_id not in self.get_feature_ids():
            raise KeyError(f"feature_id {feature_id} not added yet. Available: {self.get_feature_ids()}")
        return self._feature_ids_mapping[feature_id]['material_type_distribution']
    
    def get_material_types(self, feature_id):
        """
        Return material types for a feature.

        Parameters
        ----------
        feature_id : str
            Feature identifier.

        Returns
        -------
        list of str
            Material types.
        """
        if feature_id not in self.get_feature_ids():
            raise KeyError(f"feature_id {feature_id} not added yet. Available: {self.get_feature_ids()}")
        return self._feature_ids_mapping[feature_id]['material_type_list']
        
