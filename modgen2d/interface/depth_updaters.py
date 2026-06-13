"""
Smoothing generation utilities.

This module defines abstract and concrete classes for generating
smoothing interfaces using different smoothing techniques.
"""
from abc import ABC, abstractmethod
import numpy as np
import warnings
import modgen2d.general_functions as f
from ._read_only import DiscretizedInterfaces2DReadOnly

class AbstractDepthUpdater(ABC):
    """
    Abstract base class for updating interface depths in 2D domains.

    Subclasses must implement `update_interfaces_depths` to define
    how interface depths are initialized or modified.
    
    Parameters
    ----------
    updater_params : dict
        Dictionary containing parameters specific to the interface
        depth updater.
        
    Attributes
    ----------
    updater_params : dict
        Dictionary containing parameters specific to the interface
        depth updater.
    """
    def __init__(self, updater_params:dict):
        """
        Initialize the depth updater.
        """
        self.updater_params = updater_params
    
    @staticmethod
    def adjust_ref_x(ref_x, discretized_interface2d_instance):
        # Locate reference point in grid
        x_centers = discretized_interface2d_instance.domain.get_interface_x_centers
        
        if ref_x is None: 
            ref_idx = 0 ## Randomize here: No, avoidable rng needed.
            ref_x = x_centers[ref_idx]
        elif not isinstance(ref_x, (int, float)):
            raise ValueError("ref_x must be a number")
        elif ref_x < x_centers[0] or ref_x > x_centers[-1]:   
            edge = x_centers[0] if ref_x < x_centers[0] else x_centers[-1]
            # warn if reference point is not on grid point
            msg = f"Requested position ({ref_x:.3f}) out of domain bound. "
            msg += f"Hence, setting to closest edge/bound ({edge:.3f})."
            warnings.warn(msg)
            ref_x = edge
        return ref_x
            
    @abstractmethod #Main Purpose to not allow user to use this method directly
    def update_interfaces_depths(self, discretized_interface2d_instance, **kwargs):
        """
        Update interfaces depth at specified x_coord.

        Parameters
        ----------
        discretized_interface2d_instance
            Initial DiscretizedInterfaces2D.

        Returns
        -------
        numpy.ndarray
            Array of shape ''(nx, n_interfaces)'' containing smooth interface
            elevations.
        """
        pass

class OneBoreholeDepthUpdater(AbstractDepthUpdater):
    """
    Applies a depth update to the interface based on 1D borehole.

    Parameters
    ----------
    ref_depths : array_like
        Reference depths for each interface.
    ref_x : float, optional
        Reference x-coordinate to align depths. Defaults to first grid point if None.
    """
    def __init__(self, ref_depths, ref_x=None):
        """
        Updates interface depths using a 1D borehole reference.
        """
        updater_params = {'ref_x': ref_x}
        self.ref_depths = ref_depths
        super().__init__(updater_params)
       
    def update_interfaces_depths(self, discretized_interface2d_instance):
        """
        Shifting the interfaces based on provided ref_depths at ref_x value.

        Parameters
        ----------
        discretized_interface2d_instance
            Initial DiscretizedInterfaces2D.
        
        """
        interfaces_matrix = discretized_interface2d_instance.interfaces_matrix
        x_centers = discretized_interface2d_instance.domain.get_interface_x_centers
        n_soil_layers = discretized_interface2d_instance.n_soil_layers
        remesh_interp_method = discretized_interface2d_instance.remesh_interp_method
        ref_x = self.adjust_ref_x(self.updater_params['ref_x'], discretized_interface2d_instance)
        
        ## Make sure ref_depths are valid.
        if self.ref_depths is None:
            raise ValueError("ref_depths cannot be None for OneBoreholeDepthUpdater.")
        
        ref_depths = np.asarray(self.ref_depths, dtype=float)
        if not np.issubdtype(ref_depths.dtype, np.number):
            raise ValueError("ref_depths must contain float values")

        if ref_depths.ndim != 1 or len(ref_depths)!=n_soil_layers:
            raise ValueError ( f"The provided no of reference points for interfaces ({len(ref_depths)}) != provided no of soil layers ({n_soil_layers}). Note: Surface interface cannot be given any reference value, so have first element 0.")

        if ref_depths[0] != 0:
            raise ValueError("ref_depths must have first element 0. Surface-soil interface is auto adjusted. So, all other references are relative to surface.")
            
        if not np.all(np.diff(ref_depths) >= 0):
            raise ValueError(f"ref_depths must be monotonically increasing. Provided {ref_depths}")
    
        zs = range(n_soil_layers)
        if interfaces_matrix.shape[1] != 0:
            interp_ref_zs = f.remeshing_2D_matrix(x_old = x_centers, x_new = [ref_x],
                                                z_old = zs, z_new = zs, matrix_2d = interfaces_matrix, interp_method = remesh_interp_method)

            # computing the shift
            ref_depths+=interp_ref_zs[0,0]
            shift_z = ref_depths - interp_ref_zs[0,:]  
            shift_matrix = np.ones_like(interfaces_matrix) * shift_z[np.newaxis]
            interfaces_matrix += shift_matrix
            
        return interfaces_matrix, ref_x
        # self.set_interfaces_matrix(interfaces_matrix)
        # self._ref_x = ref_x
        
class EquidistantDepthUpdater(OneBoreholeDepthUpdater):
    """
    Sets interface depths to equally spaced intervals along the vertical span.
    
    Parameters
    ----------
    ref_x: float, optional
        Reference x-coordinate for initialization. Will save the reference point, in case merged with surface (later).
        If None, first point in the x_centers.
    
    """
    def __init__(self, ref_x=None):
        """
        Initialize the equidistant depth updater.
        """
        super().__init__(None, ref_x)
       
    def update_interfaces_depths(self, discretized_interface2d_instance):
        """
        Shifting the interfaces based on equidistant soil depths at ref_x value.

        Parameters
        ----------
        discretized_interface2d_instance
            Initial DiscretizedInterfaces2D.
        """
        
        # Replacing self.ref_depths accordingly
        span_z = discretized_interface2d_instance.domain.spans[1]
        n_soil_layers = discretized_interface2d_instance.n_soil_layers
        ref_depths = np.arange(1, n_soil_layers) * span_z / n_soil_layers
        
        # ref_depths must have first element 0. Surface-soil interface is auto adjusted. 
        # So, all other references are relative to surface.
        self.ref_depths = np.concatenate(([0.0], ref_depths))  
        return super().update_interfaces_depths(discretized_interface2d_instance)
 
class RandomDepthUpdater(OneBoreholeDepthUpdater): 
    """
    Sets interface depths randomly along the vertical span (sorted).
    
    Parameters
    ----------
    ref_x: float, optional
        Reference x-coordinate for initialization. Will save the reference point, in case merged with surface (later).
        If None, first point in the x_centers.
    
    """
    def __init__(self, ref_x=None):
        """
        Initialize the equidistant depth updater.
        """
        super().__init__(None, ref_x)
       
    def update_interfaces_depths(self, discretized_interface2d_instance):
        """
        Shifting the interfaces so that randomly generated soil depths at ref_x value.

        Parameters
        ----------
        discretized_interface2d_instance
            Initial DiscretizedInterfaces2D.
        """
        # Replacing self.ref_depths accordingly
        span_z = discretized_interface2d_instance.domain.spans[1]
        n_soil_soil_interfaces = discretized_interface2d_instance.n_soil_soil_interfaces
        rng = discretized_interface2d_instance.rng
        
        rndm_numbers = rng.random(n_soil_soil_interfaces)
        rndm_numbers.sort() 
        ref_depths = rndm_numbers * span_z
        # ref_depths must have first element 0. Surface-soil interface is auto adjusted. 
        # So, all other references are relative to surface.
        self.ref_depths = np.concatenate(([0.0], ref_depths))  
        return super().update_interfaces_depths(discretized_interface2d_instance)
       