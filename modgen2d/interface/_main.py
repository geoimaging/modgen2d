"""
Discretized 2D geological interfaces.

Defines utilities for generating, processing, remeshing,
and visualizing soil and surface interfaces over a
discretized 2D domain.
"""

from typing import Union
import warnings

import numpy as np

from modgen2d.discretized_domain2d import DiscretizedDomain2D
from ._read_only import DiscretizedInterfaces2DReadOnly
from .rough_interface_generator import AbstractRoughInterfaceGenerator
from .interface_smoother import AbstractSmoother
from .depth_updaters import AbstractDepthUpdater
import modgen2d.general_functions as f

class DiscretizedInterfaces2D(DiscretizedInterfaces2DReadOnly):
    """
    Discretized 2D soil and surface interfaces.

    Interfaces are defined on a discretized 2D domain and may represent soil–soil or surface–soil boundaries.

    Once locked, the instance becomes immutable.
    
    Parameters
    ----------
    domain : DiscretizedDomain2D
        The DiscretizedDomain2D instance describing the spans and dhs of the domain.
    n_soil_layers: int
        Number of soil layers.
    generate_surface:bool
        Whether a surface interface is present.
    roughness_multiplier : list, optional
        Roughness scaling factors per interface.
    remesh_interp_method : str, optional
        Interpolation method used during remeshing. (default: 'linear')
    rng : numpy.random.Generator, optional
        Random number generator.
        
    Attributes
    ----------
    domain : DiscretizedDomain2D
        Domain associated with the interfaces.
    interfaces_matrix : numpy.ndarray
        Interface depth matrix of shape
        ``(n_interface_x_points, n_soil_layers)``.
    n_soil_layers : int
        Number of soil layers.
    n_soil_soil_interfaces : int
        Number of soil-soil interfaces.
    generate_surface : bool
        Whether a surface interface is present.
    remesh_interp_method : str
        Interpolation method used during remeshing.
    rng : numpy.random.Generator
        Random number generator.
    """
    
    def __init__(self, domain: DiscretizedDomain2D, n_soil_layers: int, generate_surface:bool, remesh_interp_method = 'linear', rng=np.random.default_rng()):
        """
        Initializes the 'InterfaceCreator' class instance. 
        """
        super().__init__(domain, n_soil_layers, generate_surface, remesh_interp_method, rng)
    
    def apply(self, 
             command_object: 'Union[AbstractRoughInterfaceGenerator, AbstractSmoother, AbstractDepthUpdater, str]', 
            **kwargs):
        """
        Apply a command to update or modify the interfaces of the current object.

        This method executes a command based on the type of `command_object` or a string 
        identifier. The command typically updates internal state of `self`, such as 
        `interfaces_matrix`, `adj_roughness_multipliers`, or `ref_x`.
        
        Supports method chaining so multiple commands can be applied in sequence:
            self.apply(generator).apply(smoother).apply(depth_updater)

        Parameters
        ----------
        command_object : AbstractRoughInterfaceGenerator | AbstractSmoother | AbstractDepthUpdater | str
            The command to apply. Can be:
            
            - 'None' : No change
            - `AbstractRoughInterfaceGenerator` instance: generates rough interfaces and updates 
              `interfaces_matrix` and `_adj_roughness_multipliers`.
            - `AbstractSmoother` instance: smooths existing interfaces and updates `interfaces_matrix`.
            - `AbstractDepthUpdater` instance: updates interface depths and updates `interfaces_matrix` and `_ref_x`.
            - `'erosion'` or `'reverse_erosion'` : Resolves overlapping interfaces in `interfaces_matrix`.
            - `'adjust_surface_top_to_zero'`: adjusts the top surface interface to zero.
        
        **kwargs : dict
            Additional keyword arguments passed to the underlying command method.

        Returns
        -------
        self
            Returns the object itself to allow method chaining.
        
        Raises
        ------
        ValueError
            If `command_object` is not one of the supported types or string commands.

        Notes
        -----
        - This method modifies `self` in-place and does not return a value.
        - For multi-output commands (e.g., rough interface generation or depth updating), the 
          outputs are automatically applied to the corresponding attributes of `self`.
        - String commands are dispatched internally to the corresponding methods.
        """
        
        if command_object is None:
            pass
        elif isinstance(command_object, AbstractRoughInterfaceGenerator):
            interfaces_matrix, adj_roughness_multipliers = command_object.generate_rough_interfaces(self, **kwargs)
            self.set_interfaces_matrix(interfaces_matrix)
            self._adj_roughness_multipliers = adj_roughness_multipliers
            
        elif isinstance(command_object, AbstractSmoother):
            # TODO: add check to make sure interface was generated (all numbers)
            interfaces_matrix = command_object.generate_smooth_interfaces(self, **kwargs)
            self.set_interfaces_matrix(interfaces_matrix)
            
        elif isinstance(command_object, AbstractDepthUpdater):
            interfaces_matrix, ref_x = command_object.update_interfaces_depths(self, **kwargs)
            self.set_interfaces_matrix(interfaces_matrix)
            self._ref_x = ref_x
                
        elif command_object in ['erosion', 'reverse_erosion']:
            self.resolving_overlapped_interfaces(overlap_resolving_technique=command_object)
            
        elif command_object == 'adjust_surface_top_to_zero':
            self.adjust_top_of_surface_interface_to_zero()
        else:
            raise ValueError(f"Unsupported command_object: {command_object}")

        # Return self to allow chaining
        return self
    
    def apply_default_pipeline(self, rough_interface_generator:AbstractRoughInterfaceGenerator, interface_smoother:AbstractSmoother, depth_updater:AbstractDepthUpdater, overlap_resolving_technique='erosion', adjust_surface_top_to_zero=True):
        """
        Apply a sequence of interface operations to update the object's internal interfaces.

        Parameters
        ----------
        rough_interface_generator : AbstractRoughInterfaceGenerator, optional
            Generates rough interfaces and updates `interfaces_matrix` and `_adj_roughness_multipliers`.
        interface_smoother : AbstractSmoother, optional
            Smooths existing interfaces and updates `interfaces_matrix`.
        depth_updater : AbstractDepthUpdater, optional
            Updates interface depths and updates `interfaces_matrix` and `_ref_x`.
        overlap_resolving_technique : str, default='erosion'
            Method used to resolve overlapping interfaces.
        adjust_surface_top_to_zero : bool, default=True
            Whether to adjust the top surface interface to zero.
        """

        #Step 1: Generate Rough Interfaces
        #Step 2: Filter
        #Step 3: Update Interface Depth
        self.apply(rough_interface_generator).apply(interface_smoother).apply(depth_updater)
    
        # Step 4: Overlap resolution and adjust for zero
        if overlap_resolving_technique is not None:
            self.apply(overlap_resolving_technique)
            
        if adjust_surface_top_to_zero:
            self.apply('adjust_surface_top_to_zero')
            
    def replace_top_surface(self, surface_interface_instance:"DiscretizedInterfaces2D", method='pile') -> "DiscretizedInterfaces2D":
        """
        Get the interface matrix with surface included.
        ## scaling factor also replaced.
        """
        # if self.check_if_overlapping_interfaces():
        #     raise ValueError("Overlapping interfaces exist. Please use processing at main interface before adding the top surface.") 
        
        if method not in ['pile', 'erode']:
            raise ValueError(f"Methods can only be either 'pile' or 'erode'. Provided: {method}")

        if not isinstance(surface_interface_instance, DiscretizedInterfaces2D):
            raise TypeError("surface_interface_instance must be a DiscretizedInterfaces2D class or its subclass.")
        
        if not self._locked or not surface_interface_instance._locked:
            raise ValueError("Both soil and surface interface instances must be locked.")
        
        if surface_interface_instance.n_soil_layers != 1:
            raise ValueError(f"surface_interface_instance must have 1 interface (soil layer) only. Provided {surface_interface_instance.n_soil_layers}")
        
        # if np.min(surface_interface_instance.interfaces_matrix)!=0:
        #     raise ValueError(f"The surface_interface_instance must have the min value of its inteface matrix as 0. Use ._adjust_for_top_surface_interface first if needed.")
        
        if self.remesh_interp_method != surface_interface_instance.remesh_interp_method:
            raise ValueError(f"Interpolation methods of this ('{self.remesh_interp_method}') and surface_interface_instance ('{surface_interface_instance.remesh_interp_method}') does not match.")
        ## Make sure domains dhs are consistent
        if not (
            len(self.domain.spans) == len(surface_interface_instance.domain.spans)
            and all(f.is_close(a, b) for a, b in zip(self.domain.spans,
                                                surface_interface_instance.domain.spans))
        ):
            raise ValueError(
                "The domains' spans are not consistent. "
                f"Lithological domain has spans {self.domain.spans}, "
                f"while surface interface has {surface_interface_instance.domain.spans}"
            )
            
        if surface_interface_instance.domain != self.domain:
            surface_interface_instance = surface_interface_instance.remesh_interface(self.domain.dhs[0], self.domain.dhs[1])
        
        surf_interface_matrix = surface_interface_instance.interfaces_matrix
        self_interface_matrix = self.interfaces_matrix
        new_instance = self.clone()
        
        ## Adjust the scaling factor and generate_surface
        new_instance.generate_surface = surface_interface_instance.generate_surface
        roughness_multiplier = self._adj_roughness_multipliers
        
        new_scale = roughness_multiplier.copy()
        new_scale[0] = surface_interface_instance._adj_roughness_multipliers[0]
        
        if method == 'erode':
            # Preserve the ref_x value's 1D profile if not None
            if self._ref_x is not None:
                warnings.warn("Performing erosion in replace_top_surface might impact the depths_constraint (if equivalent or manual depth updater was used.)")
            
            # Perform eroding            
            new_interface_matrix[:,0] = surf_interface_matrix[:,0]
            
        else: #'pile'
            # Preserve the ref_x value's 1D profile if not None
            # 1D profile at all points are preserved when piled. so need for processing for that.
            
            # Perform piling
            surf_interface_matrix = np.ones_like(self_interface_matrix)*surf_interface_matrix
            new_interface_matrix = self_interface_matrix + surf_interface_matrix
            
        # Making the top of the surface_interface the top of the model too.
        new_instance._locked =False
        new_instance.set_interfaces_matrix(new_interface_matrix)
        new_instance.resolving_overlapped_interfaces()
        new_instance.lock_interfaces()
        return new_instance

   


    