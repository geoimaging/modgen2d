"""
Rough interface generation utilities.

This module defines abstract and concrete classes for generating
rough geological interfaces using different stochastic models
(e.g., uniform, normal, fractional Brownian motion).
"""
from abc import ABC, abstractmethod
import warnings
import numpy as np
from ._read_only import DiscretizedInterfaces2DReadOnly

class AbstractRoughInterfaceGenerator(ABC):
    """
    Abstract base class for rough interface generators.

    Concrete subclasses must implement the generate_rough_interfaces method.
    """
    def __init__(self, generator_params:dict, generate_surface:bool, roughness_multipliers:list):
        """
        Initialize the rough interface generator.

        Parameters
        ----------
        generator_params : dict
            Dictionary containing parameters specific to the interface
            generation method.
        generate_surface:bool
            Whether a surface interface is present.
        roughness_multipliers : array-like
            Scaling factor applied to each interface.
        """
        self.generator_params = generator_params
        
        if roughness_multipliers is None:
            roughness_multipliers = [int(generate_surface), 1]
            
        roughness_multipliers = np.asarray(roughness_multipliers, dtype=float)
        AbstractRoughInterfaceGenerator.check_roughness_multipliers(roughness_multipliers, generate_surface)
        
        self.roughness_multipliers = roughness_multipliers
        self.generate_surface = generate_surface
    
    def get_adjusted_roughness_multipliers(self, discretized_interface2d_instance:DiscretizedInterfaces2DReadOnly, roughness_multipliers):
        
        generate_surface = discretized_interface2d_instance.generate_surface
        n_soil_layers = discretized_interface2d_instance.n_soil_layers
        generate_surface = discretized_interface2d_instance.generate_surface
        
        if self.generate_surface != generate_surface:
            raise ValueError(f"The generate_surface flag for Rough Interface generator ({self.generate_surface}) does not match with that of discretized_interfaces2d ({generate_surface})")
        
        if len(roughness_multipliers)==1 and n_soil_layers>1 and roughness_multipliers[0]==0:
            warnings.warn(f"roughness_multipliers is [0] and hence only horizontal interfaces will be created for all {n_soil_layers} if not corrected.")
            
        adj_roughness_multipliers = np.full(n_soil_layers, roughness_multipliers[-1], dtype=float)
        min_len = min(len(roughness_multipliers), n_soil_layers)
        adj_roughness_multipliers[:min_len] = roughness_multipliers[:min_len]
        
        AbstractRoughInterfaceGenerator.check_roughness_multipliers(adj_roughness_multipliers, generate_surface)
        
        if len(adj_roughness_multipliers) != n_soil_layers:
            raise ValueError("The adjusted length of roughness_multipliers must be equal to n_soil_layers. Try re setting the multiplier.")
        
        return adj_roughness_multipliers
        
    @staticmethod
    def check_roughness_multipliers(roughness_multipliers, generate_surface):
        """
        Validate the interface scaling array.

        Parameters
        ----------
        roughness_multipliers : array-like
            One-dimensional array of scaling factors, one per interface.

        Raises
        ------
        ValueError
            If the array is not numeric, not one-dimensional,
            or is empty.
        """
        if not np.issubdtype(roughness_multipliers.dtype, np.number):
            raise ValueError("roughness_multipliers must contain float values")
        
        if roughness_multipliers.ndim != 1:
            raise ValueError("roughness_multipliers must be one-dimensional")
        
        if len(roughness_multipliers)<1:
            raise ValueError("There should be at least one number in roughness_multipliers. Found none. Set the roughness_multipliers again.")
        
        if not generate_surface and roughness_multipliers[0]!=0:
            raise ValueError(f"Models with no/hz surface must have first element on roughness_multipliers as 0. Provided {roughness_multipliers[0]}.")

        if generate_surface and roughness_multipliers[0]==0:
            raise ValueError(f"Models with surface must have first element on roughness_multipliers as non-zero. Provided {roughness_multipliers[0]}.")
        
    @abstractmethod #Main Purpose to not allow user to use this method directly
    def generate_rough_interfaces(self, discretized_interface2d_instance, **kwargs):
        """
        Generate rough interfaces.

        Parameters
        ----------
        discretized_interface2d_instance:DiscretizedInterfaces2D
            Initial DiscretizedInterfaces2D.
        **kwargs
            Generator-specific parameters.

        Returns
        -------
        numpy.ndarray
            Array of shape ''(nx, n_interfaces)'' containing interface
            elevations.
        """
        pass
    
class UniformInterfaceGen(AbstractRoughInterfaceGenerator):
    """
    Generate rough interfaces using uniformly distributed increments.
    """
    def __init__(self, max_dz_per_unit_length, generate_surface:bool, roughness_multipliers:list):
        """
        Initialize the uniform interface generator.

        Parameters
        ----------
        max_dz_per_unit_length : float
            Maximum vertical change per unit horizontal length.
        generate_surface:bool
            Whether a surface interface is present.
        roughness_multipliers : array-like
            Scaling factor applied to each interface.
        rng : numpy.random.Generator, optional
            Random number generator.
        """
        generator_params = {'max_dz_per_unit_length': max_dz_per_unit_length}
        super().__init__(generator_params, generate_surface, roughness_multipliers)
       
    def generate_rough_interfaces(self, discretized_interface2d_instance):
        """
        Generate rough interfaces using uniform random increments.

        Parameters
        ----------
        discretized_interface2d_instance:DiscretizedInterfaces2D
            Initial DiscretizedInterfaces2D.

        Returns
        -------
        numpy.ndarray
            Interface elevation matrix of shape (nx, n_interfaces).
        """
        nx, _ = discretized_interface2d_instance.interfaces_matrix.shape
        dx = discretized_interface2d_instance.domain.dhs[0]
        n_soil_layers = discretized_interface2d_instance.n_soil_layers
        interfaces_matrix = np.zeros((nx, n_soil_layers))
        rng = discretized_interface2d_instance.rng

        adj_roughness_multipliers = self.get_adjusted_roughness_multipliers(discretized_interface2d_instance, self.roughness_multipliers)
        
        base_max_dz = self.generator_params['max_dz_per_unit_length']
        z_max_change_per_dx = base_max_dz * dx

        rnd_numbers = (rng.random((nx-1, n_soil_layers)) - 0.5) * 2 #Numbers ranging from 1 and -1
        dz = rnd_numbers * z_max_change_per_dx
        dz *= adj_roughness_multipliers[:n_soil_layers]

        interfaces_matrix[1:, :] = dz
        interfaces_matrix = np.cumsum(interfaces_matrix, axis=0)

        return interfaces_matrix, adj_roughness_multipliers
         
class NormalInterfaceGen(AbstractRoughInterfaceGenerator):
    """
    Generate rough interfaces using normally distributed increments.
    """
    def __init__(self, stdev_in_unit_length, generate_surface:bool, roughness_multipliers:list):
        """
        Initialize the normal interface generator.

        Parameters
        ----------
        stdev_in_unit_length : float
            Standard deviation per unit horizontal length.
        generate_surface:bool
            Whether a surface interface is present.
        roughness_multipliers : array-like
            Scaling factor applied to each interface.
        
        """
        generator_params = {'stdev_in_unit_length': stdev_in_unit_length}
        super().__init__(generator_params, generate_surface, roughness_multipliers)
       
    def generate_rough_interfaces(self, discretized_interface2d_instance):
        """
        Generate rough interfaces using uniform random increments.

        Parameters
        ----------
        discretized_interface2d_instance:DiscretizedInterfaces2D
            Initial DiscretizedInterfaces2D.

        Returns
        -------
        numpy.ndarray
            Interface elevation matrix of shape ''(nx, n_interfaces)''.
        """
        nx, _ = discretized_interface2d_instance.interfaces_matrix.shape
        dx = discretized_interface2d_instance.domain.dhs[0]
        n_soil_layers = discretized_interface2d_instance.n_soil_layers
        interfaces_matrix = np.zeros((nx, n_soil_layers))
        rng = discretized_interface2d_instance.rng

        adj_roughness_multipliers = self.get_adjusted_roughness_multipliers(discretized_interface2d_instance, self.roughness_multipliers)

        mean = 0
        sigma_1m = self.generator_params['stdev_in_unit_length']
        sigma_dx = sigma_1m * np.sqrt(dx)  #Standard deviation grows with sqrt distance
        
        dz = rng.normal(loc=mean, scale=sigma_dx, size=(nx-1, n_soil_layers)) #Numbers ranging with mean 0
        dz *= adj_roughness_multipliers[:n_soil_layers]
            
        interfaces_matrix[1:, :] = dz
        interfaces_matrix = np.cumsum(interfaces_matrix, axis=0)
        return interfaces_matrix, adj_roughness_multipliers
