"""Abstract spatial simulation utilities for 2D lithological domains."""

import numpy as np
from abc import ABC, abstractmethod
from modgen2d.lithological_domain2d import LithologicalDomain2D
import modgen2d.general_functions as f
import warnings
import pandas as pd

class SpatialSimulator2DAbstract(ABC):
    """
    Initialize a spatial simulator.

    Parameters
    ----------
    params : dict
        Simulator-specific parameters. Every value must be a finite float or a boolean.
        An empty dictionary is allowed.
    simulated_val_for_ignored_lit_property : int, default=-99999
        Constant value assigned to ignored lithological IDs.
    rng : numpy.random.Generator, optional
        Random number generator.
    """
    def __init__(self, params, simulated_val_for_ignored_lit_property=-99999, rng=np.random.default_rng()):
        """
        Initialize a spatial simulator.
        """
        # Validate params is a dict
        invalid_keys = [
            key
            for key, value in params.items()
            if not isinstance(value, (bool, int, float, np.number))
            or not np.isfinite(value)
        ]
        
        if invalid_keys:
            raise TypeError(
                "All values in params must be finite numeric values or booleans. "
                f"Invalid parameters: {invalid_keys}."
            )
        
        self.params = {
            key: value if isinstance(value, (bool, np.bool_)) else float(value)
            for key, value in params.items()
        }

        self.simulated_val_for_ignored_lit_property = int(simulated_val_for_ignored_lit_property) #integer check in generated profiles 
        self.rng = rng
        self.reloadable = False #Only true for covariance decomposition with default (exponential) method.
        self.allow_simulation = True #False when reloaded with non-reloadable simulator.

    @abstractmethod
    def simulate(self, points, mean=0, sigma=1):
        """
        Simulate field values at given points.

        Parameters
        ----------
        points : array-like, shape (n, 2)
            List of (x, z) coordinates where the field must be evaluated.

        mean : float or sequence of 2 floats, default=0
            mean = a_m + b_m*z
            - If scalar → use this mean value for the entire field. b_m = 0
            - If sequence of 2 numbers → [a_m, b_m]

        sigma : float or None
            Standard deviation of the noise.
            - If scalar → use for entire field.
            - If None → sigma is not used in that simulator

        Returns
        -------
        numpy.ndarray, shape (n,)
            Simulated field values at the given points.
        """
        pass

    def _invalid_simulation_message(self):
        return (
            "The saved simulator could not be fully restored and/or is not "
            "currently allowed to simulate because `allow_simulation` is False. "
            "Replace it using "
            "`.change_spatial_simulator_type(new_simulator)`. "
            f"Previously saved simulator type: "
            f"{self.__class__.__name__}; "
            f"parameters: {self.params}."
        )

    def simulate_zvals_lit_profile_from_lithological_domain(self, lithologicalDomain_class:LithologicalDomain2D, gwt_depth=None, 
                                                        generate_non_spatial_profile=False, 
                                                        ignore_lithological_ids=['X']):
        """
        Simulate standardized spatial fluctuations for a lithological domain.

        Parameters
        ----------
        lithologicalDomain_class : LithologicalDomain2D
            Lithological domain definition.
        gwt_depth : float, optional
            Groundwater table depth.
        generate_non_spatial_profile : bool, default=False
            If True, generates zero-variance (non-spatial) fluctuations.
        ignore_lithological_ids : list, default=['X']
            Lithological IDs to ignore.

        Returns
        -------
        numpy.ndarray
            2D array of standardized simulated values.
        """
        if not self.allow_simulation:
            raise ValueError(self._invalid_simulation_message())
        
        #if porcessed_property_dict is None: Then simulated profiles with mean 0 and standard dev 1.

        layer_mat = lithologicalDomain_class.lithological_matrix
        xcoord = lithologicalDomain_class.domain.x_centers
        zcoord = lithologicalDomain_class.domain.z_centers
        simulated_val_for_ignored_lit_property = self.simulated_val_for_ignored_lit_property
        _ , gwt_z = np.meshgrid(xcoord, zcoord, indexing='ij')
        
        if gwt_depth is None:
            gwt_z = np.zeros_like(layer_mat, dtype=bool)
        else:
            if lithologicalDomain_class.gwt_depth is not None and gwt_depth!=lithologicalDomain_class.gwt_depth:
                raise ValueError(f"Gwt depth provided {gwt_depth} does not match with gwt_depth from lithological domain {lithologicalDomain_class.gwt_depth}.")

            gwt_z = gwt_z >= gwt_depth  # Use y or z based on your case
            
        if gwt_z.shape != layer_mat.shape:
            raise TypeError(f"The shapes does not match: {gwt_z.shape} != {layer_mat.shape}. Check x_ranges, z_ranges, layer_mat of lithological_domain2d_instance.")         
        
        # Convert matrix value into a standarized format
        vectorized_format = np.vectorize(f.format_value)
        layer_mat = vectorized_format(layer_mat)
        unique_layers = np.unique(layer_mat)
        
        simulated_pd = pd.DataFrame()
        
        for layer_id in unique_layers:
            x_idx, z_idx = np.where(layer_mat == layer_id)
            if z_idx.size == 0:
                continue
                # skip the remaining step if no matches.
            
            if layer_id in ignore_lithological_ids:
                a_m, b_m, sigma = simulated_val_for_ignored_lit_property, 0, 0
            else:
                if generate_non_spatial_profile:
                    print("Z-vals: Non-spatial-zero-sigma")
                    a_m, b_m, sigma = 0, 0, 0
                else:    
                    print("Z-vals: spatial-with-sigma")
                    a_m, b_m, sigma = 0, 0, 1
                
            print(f"Simulating z-vals for Layer ID: {layer_id}")
            
            coordinates = [[x,z] for z,x in zip(zcoord[z_idx], xcoord[x_idx])]
            simulated_pd_each = pd.DataFrame(coordinates, columns=['x', 'z'])
            simulated_pd_each['simulated_val'] = self.simulate(points=coordinates, mean=[a_m, b_m], sigma=sigma)
            simulated_pd=pd.concat([simulated_pd,simulated_pd_each], ignore_index=True)

        simulated_2d = simulated_pd.pivot(index='x', columns='z', values='simulated_val')
        simulated_2d = simulated_2d.to_numpy()
        # print(simulated_2d.to_numpy().shape)
        
        if simulated_2d.shape != layer_mat.shape:
            raise ValueError(
                f"ERROR: Pivoted matrix shape mismatch. Check if all coordinates are covered."
                f"Expected {layer_mat.shape}, got {simulated_2d.shape}"
            )
        
        return simulated_2d
   
    def simulate_profile_from_zvals_lit_profile(self, simulated_zvals_lit_profile:np.ndarray, lithologicalDomain_class:LithologicalDomain2D,
                                                  processed_property_dict:dict, gwt_depth=None, warn_inconsistent_stdev = False,
                                                  ignore_lithological_ids=['X']):
        """
        Generate a spatial property field from standardized fluctuations.

        Parameters
        ----------
        simulated_zvals_lit_profile : numpy.ndarray
            Standardized spatial fluctuations.
        lithologicalDomain_class : LithologicalDomain2D
            Lithological domain definition.
        processed_property_dict : dict
            Dictionary mapping layer IDs to their mean and stddev (wet/dry or both) properties.
        gwt_depth : float, optional
            Groundwater table depth (used for wet/dry classification).
        warn_inconsistent_stdev : bool, default=True
            Emit warnings for inconsistent variance assumptions.
        ignore_lithological_ids : list, default=['X']
            Lithological IDs to ignore.

        Returns
        -------
        numpy.ndarray
            A 2D array representing the simulated spatially correlated random field.
        """
        if not self.allow_simulation:
            raise ValueError(self._invalid_simulation_message())
        #if porcessed_property_dict is None: Then simulated profiles with mean 0 and standard dev 1.

        layer_mat = lithologicalDomain_class.lithological_matrix
        xcoord = lithologicalDomain_class.domain.x_centers
        zcoord = lithologicalDomain_class.domain.z_centers
        simulated_val_for_ignored_lit_property=self.simulated_val_for_ignored_lit_property
        
        _, z_coord_mat = np.meshgrid(xcoord, zcoord, indexing='ij')
        
        if gwt_depth is None:
            gwt_z = np.zeros_like(z_coord_mat, dtype=bool)
        else:
            if lithologicalDomain_class.gwt_depth is not None and gwt_depth!=lithologicalDomain_class.gwt_depth:
                raise ValueError(f"Gwt depth provided {gwt_depth} does not match with gwt_depth from lithological domain {lithologicalDomain_class.gwt_depth}.")

            gwt_z = z_coord_mat >= gwt_depth  # Use y or z based on your case
            
        if gwt_z.shape != layer_mat.shape:
            raise TypeError("The shapes does not match. Check x_ranges, z_ranges, layer_mat of lithological_domain2d_instance.")      
        
        if gwt_z.shape != simulated_zvals_lit_profile.shape:
            raise TypeError("The shapes does not match. Provided simulated_zvals matrix does not match with provided lithological_domain2d_instance.")      
        
        # Convert matrix value into a standarized format
        vectorized_format = np.vectorize(f.format_value)
        layer_mat = vectorized_format(layer_mat)
        unique_layers = np.unique(layer_mat)
        # print(unique_layers)
        
        mean_matrix = np.full_like(layer_mat, np.nan, dtype=float)
        mean_slope_with_depth_matrix = np.full_like(layer_mat, np.nan, dtype = float)
        stdev_matrix = np.full_like(layer_mat, np.nan, dtype = float)
        
        # Validate processed_property_dict
        processed_property_dict = f.validate_processed_property_dict(processed_property_dict)
        unique_layers = set(unique_layers)
        ignore_ids = set(ignore_lithological_ids)

        # Layers requiring data
        required_ids = unique_layers - ignore_ids

        # 1. Check missing required keys
        missing = required_ids - set(processed_property_dict)
        if missing:
            raise ValueError(f"Missing keys in processed_property_dict: {missing}")

        # 2. Check forbidden keys present
        forbidden = set(processed_property_dict) & ignore_ids
        if forbidden:
            raise KeyError(f"Forbidden (To ignore) lithological IDs present in processed_property_dict: {forbidden}")

        # # 3. Warn for extra keys #Hidden as there will always be extra in case of lithological by lithological generations.
        # extra = set(processed_property_dict) - unique_layers
        # if extra:
        #     warnings.warn(f"Extra keys in processed_property_dict ignored: {extra}")

        for layer_id in unique_layers:
            if layer_id in ignore_lithological_ids:
                
                mask = (layer_mat == layer_id)
                mean_matrix[mask]    = simulated_val_for_ignored_lit_property
                mean_slope_with_depth_matrix[mask] = 0
                stdev_matrix[mask]   = 0

                # --- Check simulated values---
                wrong_mask = simulated_zvals_lit_profile[mask] != simulated_val_for_ignored_lit_property

                if np.any(wrong_mask):
                    total = mask.sum()
                    wrong = wrong_mask.sum()
                    pct   = (wrong / total) * 100

                    raise ValueError(
                        f"Issue with provided simulated_zvals_lit_profile, for ignored lithological ID {layer_id},"
                        f" {pct:.2f}% do not match the required constant: {simulated_val_for_ignored_lit_property}."
                        f" Make sure same constant and ignored lit ID list were used during simulating z_vals."
                    )
                
            else:
                if 'both' in processed_property_dict[layer_id].keys():
                    both = processed_property_dict[layer_id]['both']
                    a_m = both['mean']  
                    b_m = both['mean_slope_with_depth']
                    sigma = both['stdev_or_cov']
                    if both['stdev_type'] == 'cov':
                        sigma*=a_m
                    
                    mask = (layer_mat == layer_id)
                    mean_matrix[mask] = a_m
                    mean_slope_with_depth_matrix[mask] = b_m
                    stdev_matrix[mask] = sigma
                    
                    if warn_inconsistent_stdev:                 
                        sim_values = simulated_zvals_lit_profile[mask]
                        if sigma == 0 and np.any(sim_values != 0):
                            warnings.warn(
                                f"Layer {layer_id}: sigma=0 but {np.sum(sim_values != 0)} z-values are non-zero."
                            )
                        elif sigma != 0 and np.all(sim_values == 0):
                            warnings.warn(
                                f"Layer {layer_id}: sigma={sigma} (i.e. non-zero) but all simulated z-values are zero."
                            )                    
                else:
                    wet = processed_property_dict[layer_id]['wet']
                    dry = processed_property_dict[layer_id]['dry']
                    
                    # Convert cov → stdev if needed
                    a_wet,  b_wet  = wet['mean'],  wet['mean_slope_with_depth']
                    a_dry,  b_dry  = dry['mean'],  dry['mean_slope_with_depth']

                    s_wet  = wet['stdev_or_cov'] *  (a_wet if  wet['stdev_type'] == 'cov' else 1)
                    s_dry  = dry['stdev_or_cov'] *  (a_dry if  dry['stdev_type'] == 'cov' else 1)

                    mask = (layer_mat == layer_id)

                    # Wet / dry masks
                    mask_wet = mask & gwt_z
                    mask_dry = mask & ~gwt_z

                    # Assign mean, mean_slope_with_depth, and stdev
                    mean_matrix[mask_wet]    = a_wet
                    mean_matrix[mask_dry]    = a_dry

                    mean_slope_with_depth_matrix[mask_wet] = b_wet
                    mean_slope_with_depth_matrix[mask_dry] = b_dry

                    stdev_matrix[mask_wet]   = s_wet
                    stdev_matrix[mask_dry]   = s_dry

                    if warn_inconsistent_stdev:
                        # Check wet portion
                        sim_values_wet = simulated_zvals_lit_profile[mask_wet]
                        if s_wet == 0 and np.any(sim_values_wet != 0):
                            warnings.warn(
                                f"Layer {layer_id} (wet): sigma=0 but {np.sum(sim_values_wet != 0)} values are non-zero."
                            )
                        elif s_wet != 0 and np.all(sim_values_wet == 0):
                            warnings.warn(
                                f"Layer {layer_id} (wet): sigma={s_wet} but all simulated values are zero."
                            )

                        # Check dry portion
                        sim_values_dry = simulated_zvals_lit_profile[mask_dry]
                        if s_dry == 0 and np.any(sim_values_dry != 0):
                            warnings.warn(
                                f"Layer {layer_id} (dry): sigma=0 but {np.sum(sim_values_dry != 0)} values are non-zero."
                            )
                        elif s_dry != 0 and np.all(sim_values_dry == 0):
                            warnings.warn(
                                f"Layer {layer_id} (dry): sigma={s_dry} but all simulated values are zero."
                            )

        if np.isnan(mean_matrix).any() or np.isnan(mean_slope_with_depth_matrix).any() or np.isnan(stdev_matrix).any():
            raise RuntimeError("NaN values remain in property matrices—processed_property_dict or ignore IDs may be inconsistent.")

        simulated_2d = (mean_matrix + mean_slope_with_depth_matrix * z_coord_mat) + simulated_zvals_lit_profile * stdev_matrix
        
        return simulated_2d

    def simulate_profile_from_lithological_domain(self, lithologicalDomain_class:LithologicalDomain2D,  
                                                  processed_property_dict=None, gwt_depth=None, warn_inconsistent_stdev=False,
                                                  ignore_lithological_ids=['X']):
        """
        Simulate a full spatial property field from a lithological domain.

        Parameters
        ----------
        lithologicalDomain_class : LithologicalDomain2D
            Lithological domain definition.
        processed_property_dict : dict, optional
            Dictionary mapping layer IDs to their mean and stddev (wet/dry or both) properties.
        gwt_depth : float, optional
            Groundwater table depth (used for wet/dry classification).
        ignore_lithological_ids : list, default=['X']
            Lithological IDs to ignore.

        Returns
        -------
        numpy.ndarray
            A 2D array representing the simulated spatially correlated random field.
        """
        if not self.allow_simulation:
            raise ValueError(self._invalid_simulation_message())
        simulated_zvals_lit_profile = self.simulate_zvals_lit_profile_from_lithological_domain(
            lithologicalDomain_class=lithologicalDomain_class, gwt_depth=gwt_depth,
            generate_non_spatial_profile=False, ignore_lithological_ids=ignore_lithological_ids)
        
        simulated_profile = self.simulate_profile_from_zvals_lit_profile(
            simulated_zvals_lit_profile, lithologicalDomain_class=lithologicalDomain_class,
            processed_property_dict=processed_property_dict, gwt_depth=gwt_depth,
            warn_inconsistent_stdev = warn_inconsistent_stdev, ignore_lithological_ids = ignore_lithological_ids)
        
        return simulated_profile
    
    @staticmethod
    def get_means_am_bm(mean):
        """
        Extract linear mean parameters.

        Parameters
        ----------
        mean : float or sequence of length 2
            Mean specification ``[a_m, b_m]`` or scalar.

        Returns
        -------
        tuple of float
            ``(a_m, b_m)``
        """
        if np.isscalar(mean):
            # Use constant mean
            a_m = mean
            b_m = 0
        else:
            # Expect list/tuple/array of length 2 → [a_m, b_m] for mean = a_m + b_m * z
            mean = np.asarray(mean)
            if mean.size != 2:
                raise ValueError("mean must be a scalar or a sequence of length 2")
            a_m = mean[0]
            b_m = mean[1]
        return a_m, b_m
    
    @staticmethod
    def check_points(points):
        """
        Validate and normalize input coordinates.

        Parameters
        ----------
        points : array-like, shape (n, 2)
            Coordinate array.

        Returns
        -------
        numpy.ndarray
            Validated coordinate array.
        """
        # Convert to array
        pts = np.asarray(points)

        # Must be an array of shape (N, 2)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("points must be an (N, 2) numeric array.")

        # Must be numeric
        if not np.issubdtype(pts.dtype, np.number):
            raise TypeError("points array must contain only numeric values (int or float).")

        return pts
    
    @property
    def get_config(self):
        """
        Export simulator configuration.

        Returns
        -------
        dict
            Serializable simulator configuration.
        """
        return {
            'params': self.params,
            'simulated_val_for_ignored_lit_property': self.simulated_val_for_ignored_lit_property,
            'rng_state': self.rng.bit_generator.state,
            'reloadable': self.reloadable,
            'allow_simulation': self.allow_simulation,
            'simulator_type_name':self.__class__.__name__
        }
        
    @classmethod
    def from_config(cls, config_dict):
        """
        Reconstruct a simulator from a configuration dictionary.

        Parameters
        ----------
        config_dict : dict
            Simulator configuration.

        Returns
        -------
        SpatialSimulator2D
            Reconstructed simulator instance.
        """
        if not isinstance(config_dict, dict):
            raise TypeError("Expected a dictionary.")
        try:
            # Support both the current nested format and the previous flat format.
            if "params" in config_dict:
                params = config_dict["params"]
            
                if not isinstance(params, dict):
                    raise TypeError("config_dict['params'] must be a dictionary.")
            
                if any(key in config_dict for key in ("theta_x", "theta_z")):
                    raise ValueError(
                        "theta_x and theta_z cannot be provided both inside "
                        "'params' and at the top level."
                    )
            
                params = params.copy()
            
            else:
                params = {}
            
                if "theta_x" in config_dict:
                    params["theta_x"] = config_dict["theta_x"]
            
                if "theta_z" in config_dict:
                    params["theta_z"] = config_dict["theta_z"]

            reloadable = config_dict.get('reloadable', False)    
            allow_simulation = config_dict.get('allow_simulation', False)    
            
            simulated_val_for_ignored_lit_property = config_dict['simulated_val_for_ignored_lit_property']
            rng = np.random.default_rng()
            rng.bit_generator.state = config_dict['rng_state']
            obj = cls.__new__(cls) #Note cannot be used with ABC but works with any subclasses.
            obj.params = params
            obj.simulated_val_for_ignored_lit_property = simulated_val_for_ignored_lit_property
            obj.rng = rng
            obj.reloadable = reloadable
            obj.allow_simulation = reloadable and allow_simulation

            # expected = cls.__name__
            # actual = config_dict.get('simulator_type_name')
            # if obj.__class__.__name__ != config_dict['simulator_type_name']:
            #     warnings.warn(f"Loading simulator as '{expected}' but config was saved from '{actual}'. Use  .change_spatial_simulator_type({actual})",
            #     RuntimeWarning
            # )

            if not (obj.reloadable and obj.allow_simulation):
                warnings.warn(
                    self._invalid_simulation_message(),
                    RuntimeWarning,
                    stacklevel=2,
                )
            return obj
        
        except (KeyError, TypeError) as e:
            raise ValueError(f"Invalid config dictionary: {e}")   
        
        # fOR EQUAL CHECK.. CHECK THE TYPE TOO.
        # FOR LATER SAVE; CHANGE TYPE IF NEEDED.
        
