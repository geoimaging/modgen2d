# This file is part of modgen2d a Python package for ...
# Copyright (C) XXXX Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
# LICENSE

"""Global configuration for two-dimensional soil interfaces."""
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
import numpy as np
from ._read_only import DiscretizedInterfaces2DReadOnly
from modgen2d.meta_class import _StrictProtectedMeta, _internal_classmethod, classproperty

class GlobalSoilInterfaceConfig(metaclass=_StrictProtectedMeta):
    """
    Global configuration manager for the active/current soil interface model instance and 
    its processing behavior.
    
    This class acts as a centralized registry that stores the current interface, 
    processing mode, and a revision token used to detect boundary changes.
    """

    __slots__ = []  #Does not allow any instance variables. Only class variables.
    
    # central definition of DEFAULTS
    _DEFAULTS = {
        "_discretized_interface2d_instance": None,
        "_revision_id": 0,
    }
    
    # initialize class attributes once
    # default values
    _discretized_interface2d_instance = None
    _revision_id = 0
        
    @_internal_classmethod
    def reset(cls):
        """Reset global configuration to default values."""
        for key, val in cls._DEFAULTS.items():
            setattr(cls, key, val)
    
    @_internal_classmethod
    def set_soil_interface(cls, discretized_interface2d_instance:DiscretizedInterfaces2DReadOnly, 
                           force_set=False):
        """
        Set the global soil interface configuration. 
        
        It generates a _revision_id, which is a randomly generated integer that uniquely identifies the 
        current configuration state. Updated every time `set_surface_interface()` is called. 
        Allows downstream systems to detect changes in current/active interface2d instance.

        Parameters
        ----------
        discretized_interface2d_instance : DiscretizedInterfaces2D
            The interface2d instance to activate globally.
        
        force_set : bool, default=False
            If False and a surface interface is already set, a RuntimeError 
            will be raised. If True, the existing interface is overwritten.
       
        Raises
        ------
        RuntimeError
            If an interface is already set and ``force_set=False``.
        TypeError
            If input is invalid.
        """    
        
        if cls.get_revision_id() != 0 and not force_set:
            raise RuntimeError(
                "Surface interface already set. "
                "Use force-set=True if you intentionally want to overwrite it.")
        
        if discretized_interface2d_instance is None:
            raise TypeError("soil_interface2d_instance cannot be None")
        
        if not isinstance(discretized_interface2d_instance, DiscretizedInterfaces2DReadOnly):
            raise TypeError("discretized_interface2d_instance must be from subclass of DiscretizedInterfaces2D class.")
        
        discretized_interface2d_instance._locked = False
        discretized_interface2d_instance.lock_interfaces()
        
        cls._discretized_interface2d_instance = discretized_interface2d_instance
        
        low = 1
        high = 2**63    # exclusive upper bound
        magnitude = np.random.randint(low, high, dtype=np.int64)
        sign = 1 if np.random.random() < 0.5 else -1
        unique_code = magnitude * sign
        cls._revision_id  = int(unique_code) 
            
    @_internal_classmethod
    def get_revision_id(cls):
        """
        Return the current revision identifier.

        Returns
        -------
        int
            Revision ID.
        """
        return cls._revision_id
    
    @_internal_classmethod
    def get_interface_instance(cls):
        return cls._discretized_interface2d_instance

    @_internal_classmethod    
    def get_config_status(cls, previous_revision_id):
        """
        Check whether an external module's cached configuration is still valid.
        
        Parameters
        ----------
        previous_revision_id : int
            The revision ID previously recorded by the caller.
        
        Returns
        -------
        Boolean
            True  : Fully consistent — same revision, same compute mode.
            False : Revision changed
        """
        current_revision_id = cls.get_revision_id()
        return previous_revision_id == current_revision_id
    
    @classproperty
    def get_config(cls):
        self_config = {}
        self_config['_discretized_interface2d_instance'] =  cls.get_interface_instance().get_config
        self_config['_revision_id'] = cls.get_revision_id()
        return self_config

    @_internal_classmethod
    def set_revision_id(cls, revision_id, force_set=False):
        """
        Set revision ID explicitly.

        Notes
        -----
        Intended only for use with ``from_config``.
        """
        if force_set is False:
            raise ValueError("Note: This method is only intended for using .from_config. Not recommended for any other uses. set force_set to True for using.")
        cls._revision_id = revision_id
    
    @classmethod
    def set_soil_interface_from_config(cls, config_dict):
        """
        Restore configuration from a serialized dictionary.

        Parameters
        ----------
        config_dict : dict
            Serialized configuration.

        Raises
        ------
        ValueError
            If configuration is invalid.
        """
        if not isinstance(config_dict, dict):
            raise TypeError("Expected a dictionary.")
        try:
            discretized_interface2d_instance = DiscretizedInterfaces2DReadOnly.from_config(config_dict['_discretized_interface2d_instance'])
            cls.set_soil_interface(discretized_interface2d_instance, force_set=True)
            cls.set_revision_id(config_dict['_revision_id'], True)

        except (KeyError, TypeError) as e:
            raise ValueError(f"Invalid config dictionary: {e}")
    
    def __eq__(self, other):
        if not isinstance(other, GlobalSoilInterfaceConfig):
            return NotImplemented
        
        # units_check = self.units_config == other.units_config
        # spans_check = np.allclose(self._spans_in_domain_len_units, other._spans_in_domain_len_units)
        # dhs_check = np.allclose(self._dhs_in_domain_len_units, other._dhs_in_domain_len_units)
        # return (
        #     units_check
        #     and spans_check
        #     and dhs_check
        # )
