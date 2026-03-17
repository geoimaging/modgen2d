"""
Smoothing generation utilities.

This module defines abstract and concrete classes for generating
smoothing interfaces using different smoothing techniques.
"""
from abc import ABC, abstractmethod
from scipy.signal import savgol_filter

class AbstractSmoother(ABC):
    """
    Abstract base class for smoothing interfaces.

    Concrete subclasses must implement the generate_smooth_interfaces method.
    """
    def __init__(self, generator_params:dict):
        """
        Initialize the rough interface generator.

        Parameters
        ----------
        generator_params : dict
            Dictionary containing parameters specific to the interface
            smoothing method.
        rng : numpy.random.Generator, optional
            Random number generator used for stochastic sampling.
        """
        self.generator_params = generator_params
    
    @abstractmethod #Main Purpose to not allow user to use this method directly
    def generate_smooth_interfaces(self, discretized_interface2d_instance, **kwargs):
        """
        Generate smooth interfaces.

        Parameters
        ----------
        discretized_interface2d_instance:DiscretizedInterfaces2D
            Initial DiscretizedInterfaces2D.
       
        Returns
        -------
        numpy.ndarray
            Array of shape ''(nx, n_interfaces)'' containing smooth interface
            elevations.
        """
        pass
    
class SavGol2DSmoother(AbstractSmoother):
    """
    Applies a Savitzky-Golay filter to smooth the interface.

    """
    def __init__(self, filter_window_length=21, filter_polyorder=3):
        """
        Initialize the uniform interface generator.

        Parameters
        ----------
        filter_window_length: int
            Window size for the filter. If the value is zero, then it means no filtering.
        filter_polyorder: int
            Polynomial order for the filter
        """
        generator_params = {'filter_window_length': filter_window_length,
                            'filter_polyorder':filter_polyorder,
                            }
        super().__init__(generator_params)
       
    def generate_smooth_interfaces(self, discretized_interface2d_instance):
        """
        Generate rough interfaces using uniform random increments.

        Parameters
        ----------
        discretized_interface2d_instance:DiscretizedInterfaces2D
            Initial DiscretizedInterfaces2D.
       
            
        Returns
        -------
        numpy.ndarray
            Interface elevation matrix of same shape as init_interfaces_matrix.
        """
        init_interfaces_matrix = discretized_interface2d_instance.interfaces_matrix
        if self.generator_params['filter_window_length']!=0:
            interfaces_matrix = savgol_filter(
                init_interfaces_matrix, 
                window_length=self.generator_params['filter_window_length'], 
                polyorder=self.generator_params['filter_polyorder'], axis=0)#, window_length, polyorder

        return interfaces_matrix