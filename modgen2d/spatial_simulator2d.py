"""Spatial simulation utilities for 2D lithological domains."""

import numpy as np
from abc import ABC, abstractmethod
from modgen2d.generated_model2d import GeneratedModel2D
from modgen2d.lithological_domain2d import LithologicalDomain2D
import modgen2d.general_functions as f
import warnings
import pandas as pd

class SpatialSimulator2D(ABC):
    def __init__(self, theta_x, theta_z, simulated_val_for_ignored_lit_property=-99999, rng=np.random.default_rng()):
        """
        Initialize a spatial simulator.

        Parameters
        ----------
        theta_x : float or None
            Correlation length in the x-direction.
        theta_z : float or None
            Correlation length in the z-direction.
        simulated_val_for_ignored_lit_property : int, default=-99999
            Constant value assigned to ignored lithological IDs.
        rng : numpy.random.Generator, optional
            Random number generator.
        """
        # Validate theta_x and theta_z
        if theta_x is not None and not isinstance(theta_x, (int, float)):
            raise TypeError("theta_x must be float or None.")
        if theta_z is not None and not isinstance(theta_z, (int, float)):
            raise TypeError("theta_z must be float or None.")

        self.theta_x = theta_x
        self.theta_z = theta_z
        self.simulated_val_for_ignored_lit_property = int(simulated_val_for_ignored_lit_property) #integer check in generated profiles 
        self.rng = rng

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
    
    def change_spatial_simulator_type(self, new_simulator_class):
        """
        Convert the simulator to another simulator type.

        The internal state and RNG are preserved.

        Parameters
        ----------
        new_simulator_class : type
            Subclass of :class:`SpatialSimulator2D`.

        Returns
        -------
        SpatialSimulator2D
            New simulator instance of the requested type.
        """
        if not issubclass(new_simulator_class, SpatialSimulator2D):
            raise TypeError(
                f"{new_simulator_class.__name__} is not a SpatialSimulator2D"
            )

        # clone RNG deterministically
        rng = np.random.default_rng()
        rng.bit_generator.state = self.rng.bit_generator.state

        # bypass __init__
        obj = new_simulator_class.__new__(new_simulator_class)

        # copy common state
        obj.theta_x = self.theta_x
        obj.theta_z = self.theta_z
        obj.simulated_val_for_ignored_lit_property = self.simulated_val_for_ignored_lit_property
        obj.rng = rng

        return obj
    
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
                                                  processed_property_dict:dict, gwt_depth=None, warn_inconsistent_stdev = True,
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
        mean_bm_matrix = np.full_like(layer_mat, np.nan, dtype = float)
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

        # 3. Warn for extra keys
        extra = set(processed_property_dict) - unique_layers
        if extra:
            warnings.warn(f"Extra keys in processed_property_dict ignored: {extra}")

        for layer_id in unique_layers:
            if layer_id in ignore_lithological_ids:
                
                mask = (layer_mat == layer_id)
                mean_matrix[mask]    = simulated_val_for_ignored_lit_property
                mean_bm_matrix[mask] = 0
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
                    b_m = both['mean_bm']
                    sigma = both['stdev_or_cov']
                    if both['stdev_type'] == 'cov':
                        sigma*=a_m
                    
                    mask = (layer_mat == layer_id)
                    mean_matrix[mask] = a_m
                    mean_bm_matrix[mask] = b_m
                    stdev_matrix[mask] = sigma
                    
                    if warn_inconsistent_stdev:                 
                        sim_values = simulated_zvals_lit_profile[mask]
                        if sigma == 0 and np.any(sim_values != 0):
                            warnings.warn(
                                f"Layer {layer_id}: sigma=0 but {np.sum(sim_values != 0)} values are non-zero."
                            )
                        elif sigma != 0 and np.all(sim_values == 0):
                            warnings.warn(
                                f"Layer {layer_id}: sigma={sigma} but all simulated values are zero."
                            )                    
                else:
                    wet = processed_property_dict[layer_id]['wet']
                    dry = processed_property_dict[layer_id]['dry']
                    
                    # Convert cov → stdev if needed
                    a_wet,  b_wet  = wet['mean'],  wet['mean_bm']
                    a_dry,  b_dry  = dry['mean'],  dry['mean_bm']

                    s_wet  = wet['stdev_or_cov'] *  (a_wet if  wet['stdev_type'] == 'cov' else 1)
                    s_dry  = dry['stdev_or_cov'] *  (a_dry if  dry['stdev_type'] == 'cov' else 1)

                    mask = (layer_mat == layer_id)

                    # Wet / dry masks
                    mask_wet = mask & gwt_z
                    mask_dry = mask & ~gwt_z

                    # Assign mean, mean_bm, and stdev
                    mean_matrix[mask_wet]    = a_wet
                    mean_matrix[mask_dry]    = a_dry

                    mean_bm_matrix[mask_wet] = b_wet
                    mean_bm_matrix[mask_dry] = b_dry

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

        if np.isnan(mean_matrix).any() or np.isnan(mean_bm_matrix).any() or np.isnan(stdev_matrix).any():
            raise RuntimeError("NaN values remain in property matrices—processed_property_dict or ignore IDs may be inconsistent.")

        simulated_2d = (mean_matrix + mean_bm_matrix * z_coord_mat) + simulated_zvals_lit_profile * stdev_matrix
        
        return simulated_2d

    def simulate_profile_from_lithological_domain(self, lithologicalDomain_class:LithologicalDomain2D,  
                                                  processed_property_dict=None, gwt_depth=None,
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
        simulated_zvals_lit_profile = self.simulate_zvals_lit_profile_from_lithological_domain(
            lithologicalDomain_class=lithologicalDomain_class, gwt_depth=gwt_depth,
            generate_non_spatial_profile=False, ignore_lithological_ids=ignore_lithological_ids)
        
        simulated_profile = self.simulate_profile_from_zvals_lit_profile(
            simulated_zvals_lit_profile, lithologicalDomain_class=lithologicalDomain_class,
            processed_property_dict=processed_property_dict, gwt_depth=gwt_depth,
            warn_inconsistent_stdev = True, ignore_lithological_ids = ignore_lithological_ids)
        
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
            'theta_x': self.theta_x,
            'theta_z': self.theta_z,
            'simulated_val_for_ignored_lit_property': self.simulated_val_for_ignored_lit_property,
            'rng_state': self.rng.bit_generator.state,
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
            theta_x, theta_z = config_dict['theta_x'], config_dict['theta_z']
            simulated_val_for_ignored_lit_property = config_dict['simulated_val_for_ignored_lit_property']
            rng = np.random.default_rng()
            rng.bit_generator.state = config_dict['rng_state']
            obj = cls.__new__(cls) #Note cannot be used with ABC but works with any subclasses.
            obj.theta_x = theta_x
            obj.theta_z = theta_z
            obj.simulated_val_for_ignored_lit_property = simulated_val_for_ignored_lit_property
            obj.rng = rng
            
            expected = cls.__name__
            actual = config_dict.get('simulator_type_name')
            if obj.__class__.__name__ != config_dict['simulator_type_name']:
                warnings.warn(f"Loading simulator as '{expected}' but config was saved from '{actual}'. Use  .change_spatial_simulator_type({actual})",
                RuntimeWarning
            )
            
            return obj
        
        except (KeyError, TypeError) as e:
            raise ValueError(f"Invalid config dictionary: {e}")   
        
        # fOR EQUAL CHECK.. CHECK THE TYPE TOO.
        # FOR LATER SAVE; CHANGE TYPE IF NEEDED.
        
class ConstantSimulator(SpatialSimulator2D):
    """
    Deterministic non-spatial simulator.

    Produces values using only the specified mean function.
    """
    def __init__(self, simulated_val_for_ignored_lit_property=-99999, rng=np.random.default_rng()):
        super().__init__(None, None, simulated_val_for_ignored_lit_property, rng)
    
    def simulate(self, points, mean=0, sigma=None):
        a_m, b_m = self.get_means_am_bm(mean)    
        pts = self.check_points(points)
        z = pts[:, 1]      # extract z column
        simulated_f = a_m + b_m * z   # mean function: m(z) = a_m + b_m * z
        return simulated_f

## All in One (Detailed demo in Random_Field_simulation_v1 and v2::)
class CovarianceDecompositionSimulator(SpatialSimulator2D):
    """
    Spatial simulator using covariance decomposition.

    Generates Gaussian random fields using exponential correlation
    functions and Cholesky decomposition.
    """
    def __init__(self, theta_x, theta_z, simulated_val_for_ignored_lit_property=-99999, rng=np.random.default_rng()):
        super().__init__(theta_x, theta_z, simulated_val_for_ignored_lit_property, rng)
        self.default_simulator_type=True  #To use in loading gen_model_collection from config, so dont use True for any other case.
    
    def _compute_correlation_matrix(self, points):
        """
        Compute the Cholesky factor of correlation matrix R. Note: C = sigma^2 * R,
        where R is the correlation matrix determined by theta_x/theta_z.
        Returns L such that C = L @ L.T
        """
        pts = self.check_points(points)
        
        x = pts[:, 0][:, None]  # (N,1)
        z = pts[:, 1][:, None]  # (N,1)

        # Pairwise separations
        dx = np.abs(x - x.T)
        dz = np.abs(z - z.T)

        # Correlation matrix (σ=1)
        R = np.exp(-2 * dx / self.theta_x) * np.exp(-2 * dz / self.theta_z)

        # Cholesky of correlation matrix
        L_R = np.linalg.cholesky(R)
        
        return L_R

    def simulate(self, points, mean=0, sigma=1):
        """
        mean : float or sequence of 2 floats, default=0
            mean = a_m + b_m*z
            - If scalar → use this mean value for the entire field. b_m = 0
            - If sequence of 2 numbers → [a_m, b_m]
            
        sigma : float
            Standard deviation scaling for the field.
        """
        # Step 1: Compute linear mean trend
        pts = self.check_points(points)
        
        a_m, b_m = self.get_means_am_bm(mean)
        z = pts[:, 1]
        mean_vector = a_m + b_m * z
        
        if sigma==0:
            return mean_vector
        
        # Step 2: Compute Cholesky of correlation/covariance
        L = self._compute_correlation_matrix(pts)  # shape (N,N)

        # Step 3: Generate standard normal vector
        u = self.rng.standard_normal(pts.shape[0])   # shape (N,)

        # Step 4: Multiply to get correlated deviations
        correlated_field = L @ u                        # shape (N,)

        # Step 5: Scale by sigma and add mean trend
        #simulated_vector =  Mean + correlated_field * sigma
        simulated_field = mean_vector + correlated_field * sigma
        return simulated_field
    
def check_for_zero_sigma(processed_property_dict_layer_id):
    """
    Check if the layer_id has zero stdev_or_cov value in all its possible case.
        {'wet': {'mean': float, 'mean_bm': float (optional), 'stdev_or_cov': float, 'stdev_type':string}, 
         'dry': {'mean': float, 'mean_bm': float (optional), 'stdev_or_cov': float, 'stdev_type':string}},
    Or,
        {'both': {'mean': float, 'mean_bm': float (optional), 'stdev_or_cov': float, 'stdev_type':string}},                      

    Parameters:
    processed_property_dict : dict
        Dictionary wet/dry/both and their corresponding mean and standard deviation values as nested dictionaries.
    
    Returns:
    True if zero stdev_or_cov in all cases, else False
    """
    
    # Validate the dictionary through validate_processed_property_dict
    keys = set(processed_property_dict_layer_id.keys())
    optional_key = 'layer0_air'
    keys.discard(optional_key)

    valid_sets = [{'wet', 'dry'}, {'both'}]
    assert keys in valid_sets, f"Invalid key combination: {keys}"   ## Though already in more detail validated using validate_processed_property_dict
     
    for k in keys:
        stdev_val = processed_property_dict_layer_id[k]['stdev_or_cov']
        if stdev_val!=0:
            return False
    return True

