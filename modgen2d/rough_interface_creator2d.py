"""
Rough interface generation utilities.

This module defines abstract and concrete classes for generating
rough geological interfaces using different stochastic models
(e.g., uniform, normal, fractional Brownian motion).
"""
from abc import ABC, abstractmethod

import numpy as np
from fbm import FBM

class AbstractRoughInterfaceCreator(ABC):
    """
    Abstract base class for rough interface generators.

    Concrete subclasses must implement the generate_rough_interfaces method.
    """
    def __init__(self, generator_params:dict, rng=np.random.default_rng()):
        """
        Initialize the rough interface generator.

        Parameters
        ----------
        generator_params : dict
            Dictionary containing parameters specific to the interface
            generation method.
        rng : numpy.random.Generator, optional
            Random number generator used for stochastic sampling.
        """
        self.generator_params = generator_params
        self.rng = rng
    
    @staticmethod
    def check_rough_interface_generator_scale(rough_interface_generator_scale):
        """
        Validate the interface scaling array.

        Parameters
        ----------
        rough_interface_generator_scale : array-like
            One-dimensional array of scaling factors, one per interface.

        Raises
        ------
        ValueError
            If the array is not numeric, not one-dimensional,
            or is empty.
        """
        if not np.issubdtype(rough_interface_generator_scale.dtype, np.number):
            raise ValueError("rough_interface_generator_scale must contain float values")
        
        if rough_interface_generator_scale.ndim != 1:
            raise ValueError("rough_interface_generator_scale must be one-dimensional")
        
        if len(rough_interface_generator_scale)<1:
            raise ValueError("There should be at least one number in rough_interface_generator_scale. Found none. Set the rough_interface_generator_scale again.")
        
    @abstractmethod #Main Purpose to not allow user to use this method directly
    def generate_rough_interfaces(self, rough_interface_generator_scale, nx, dx, **kwargs):
        """
        Generate rough interfaces.

        Parameters
        ----------
        rough_interface_generator_scale : array-like
            Scaling factor applied to each interface.
        nx : int
            Number of discretization points along the horizontal direction.
        dx : float or None
            Horizontal discretization size. May be ignored by some generators.
        **kwargs
            Generator-specific parameters.

        Returns
        -------
        numpy.ndarray
            Array of shape ''(nx, n_interfaces)'' containing interface
            elevations.
        """
        pass
    
class UniformInterfaceGen(AbstractRoughInterfaceCreator):
    """
    Generate rough interfaces using uniformly distributed increments.
    """
    def __init__(self, max_dz_per_unit_length, rng=np.random.default_rng()):
        """
        Initialize the uniform interface generator.

        Parameters
        ----------
        max_dz_per_unit_length : float
            Maximum vertical change per unit horizontal length.
        rng : numpy.random.Generator, optional
            Random number generator.
        """
        generator_params = {'max_dz_per_unit_length': max_dz_per_unit_length}
        super().__init__(generator_params, rng)
       
    def generate_rough_interfaces(self, rough_interface_generator_scale, nx, dx):
        """
        Generate rough interfaces using uniform random increments.

        Parameters
        ----------
        rough_interface_generator_scale : array-like
            Scaling factor for each interface.
        nx : int
            Number of horizontal discretization points.
        dx : float
            Horizontal discretization size.

        Returns
        -------
        numpy.ndarray
            Interface elevation matrix of shape (nx, n_interfaces).
        """
        rough_interface_generator_scale = np.asarray(rough_interface_generator_scale, dtype=float)
        self.check_rough_interface_generator_scale(rough_interface_generator_scale)
        n_soil_layers = len(rough_interface_generator_scale)
        
        interfaces_matrix = np.zeros((nx, n_soil_layers))
    
        base_max_dz = self.generator_params['max_dz_per_unit_length']
        z_max_change_per_dx = base_max_dz * dx

        rnd_numbers = (self.rng.random((nx-1, n_soil_layers)) - 0.5) * 2 #Numbers ranging from 1 and -1
        dz = rnd_numbers * z_max_change_per_dx
        dz *= rough_interface_generator_scale[:n_soil_layers]


        interfaces_matrix[1:, :] = dz
        interfaces_matrix = np.cumsum(interfaces_matrix, axis=0)

        return interfaces_matrix
         
class NormalInterfaceGen(AbstractRoughInterfaceCreator):
    """
    Generate rough interfaces using normally distributed increments.
    """
    def __init__(self, stdev_in_unit_length, rng=np.random.default_rng()):
        """
        Initialize the normal interface generator.

        Parameters
        ----------
        stdev_in_unit_length : float
            Standard deviation per unit horizontal length.
        rng : numpy.random.Generator, optional
            Random number generator.
        """
        generator_params = {'stdev_in_unit_length': stdev_in_unit_length}
        super().__init__(generator_params, rng)
       
    def generate_rough_interfaces(self, rough_interface_generator_scale, nx, dx):
        """
        Generate rough interfaces using Gaussian random increments.

        Parameters
        ----------
        rough_interface_generator_scale : array-like
            Scaling factor for each interface.
        nx : int
            Number of horizontal discretization points.
        dx : float
            Horizontal discretization size.

        Returns
        -------
        numpy.ndarray
            Interface elevation matrix of shape ''(nx, n_interfaces)''.
        """
        rough_interface_generator_scale = np.asarray(rough_interface_generator_scale, dtype=float)
        self.check_rough_interface_generator_scale(rough_interface_generator_scale)
        
        n_soil_layers = len(rough_interface_generator_scale)
        interfaces_matrix = np.zeros((nx, n_soil_layers))

        mean = 0
        sigma_1m = self.generator_params['stdev_in_unit_length']
        sigma_dx = sigma_1m * np.sqrt(dx)  #Standard deviation grows with sqrt distance
        
        dz = self.rng.normal(loc=mean, scale=sigma_dx, size=(nx-1, n_soil_layers)) #Numbers ranging with mean 0
        dz *= rough_interface_generator_scale[:n_soil_layers]
            
        interfaces_matrix[1:, :] = dz
        interfaces_matrix = np.cumsum(interfaces_matrix, axis=0)
        return interfaces_matrix
    
class FBMInterfaceGen(AbstractRoughInterfaceCreator):
    """
    Generate rough interfaces using fractional Brownian motion (fBM).
    """
    def __init__(self, H, length, method, rng=np.random.default_rng()):
        """
        Initialize the fractional Brownian motion interface generator.

        Parameters
        ----------
        H : float
            Hurst exponent controlling roughness.
        length : float
            Total horizontal length of the interface.
        method : str
            fBM generation method supported by ''fbm.FBM''.
        rng : numpy.random.Generator, optional
            Random number generator.
        """
        generator_params = {'H': H,
                            'length': length,
                            'method': method}
        super().__init__(generator_params, rng)
       
    def generate_rough_interfaces(self, rough_interface_generator_scale, nx, dx='ignored'):
        """
        Generate rough interfaces using fractional Brownian motion.

        Parameters
        ----------
        rough_interface_generator_scale : array-like
            Scaling factor for each interface.
        nx : int
            Number of horizontal discretization points.
        dx : ignored
            Included for API consistency; not used.

        Returns
        -------
        numpy.ndarray
            Interface elevation matrix of shape ''(nx, n_interfaces)''.
        """
        rough_interface_generator_scale = np.asarray(rough_interface_generator_scale, dtype=float)
        self.check_rough_interface_generator_scale(rough_interface_generator_scale)
        
        n_soil_layers = len(rough_interface_generator_scale)
        interfaces_matrix = np.zeros((nx, n_soil_layers))

        H = self.generator_params['H']
        L = self.generator_params['length'] # *surface_scaling_factor While this gives approx scaling
        method = self.generator_params['method']
    
        n = nx - 1
        for j in range(n_soil_layers):
            scale = rough_interface_generator_scale[j]
            #generates n+1 data ie n increments
            rnd_layer = FBM(n=n, hurst=H, length=L, method=method).fbm() * scale  
            interfaces_matrix[:, j]= rnd_layer
        return interfaces_matrix
    
    