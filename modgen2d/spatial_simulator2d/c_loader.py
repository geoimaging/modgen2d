from .b_constant import ConstantSimulator
from .b_covdec import CovarianceDecompositionSimulator
from .b_gstools import GSToolsSimulator

import warnings
import numpy as np

def load_simulator_from_config(config_dict):
    if not isinstance(config_dict, dict):
        raise TypeError("Expected a dictionary.")
    try:
        sim_type = config_dict['simulator_type_name']
        simulated_val_for_ignored_lit_property = config_dict['simulated_val_for_ignored_lit_property']
        rng = np.random.default_rng()
        rng.bit_generator.state = config_dict['rng_state']
    
        # Support both the current nested format for params and the previous flat format (thetas outside).
        if "params" in config_dict:
            params = config_dict["params"]
        
            if not isinstance(params, dict):
                raise TypeError("config_dict['params'] must be a dictionary.")
        
            if any(key in config_dict for key in ("theta_x", "theta_z")):
                raise ValueError(
                    "theta_x and theta_z cannot be provided both inside "
                    "'params' and at the top level."
                )
            allow_simulation = config_dict.get('allow_simulation', False)  
            
        else:  #Old version 
            params = {}
        
            if config_dict.get("theta_x") is not None:
                params["theta_x"] = config_dict["theta_x"]
            
            if config_dict.get("theta_z") is not None:
                params["theta_z"] = config_dict["theta_z"]
                
            allow_simulation = sim_type in ['ConstantSimulator', 'CovarianceDecompositionSimulator']

        processed_config_dict = {
            'params': params,
            'simulated_val_for_ignored_lit_property': simulated_val_for_ignored_lit_property,
            'rng_state': config_dict['rng_state'],
            'allow_simulation': allow_simulation,
            'simulator_type_name':sim_type
        }
        
        if sim_type == "ConstantSimulator":
            return ConstantSimulator.from_config(processed_config_dict)
        if sim_type == "CovarianceDecompositionSimulator":
            return CovarianceDecompositionSimulator.from_config(processed_config_dict)
        if sim_type == "GSToolsSimulator":
            return GSToolsSimulator.from_config(processed_config_dict)
    
        # In case different sim_type, dont allow simulation.
        obj = ConstantSimulator(simulated_val_for_ignored_lit_property=simulated_val_for_ignored_lit_property, rng=rng)
        obj.allow_simulation = False

        msg = (
            "The saved simulator could not be fully restored and/or is not "
            "currently allowed to simulate because `allow_simulation` is False. "
            "Replace it using "
            "`.change_spatial_simulator_type(new_simulator)`. "
            f"Previously saved simulator type: {sim_type}; "
            f"parameters: {params}; "
            f"simulated_val_for_ignored_lit_property: {simulated_val_for_ignored_lit_property}; "
        )

        if not (obj.allow_simulation):
            warnings.warn(
                msg,
                RuntimeWarning,
                stacklevel=2,
            )
        return obj
        
    except (KeyError, TypeError) as e:
        raise ValueError(f"Invalid config dictionary: {e}")   
    