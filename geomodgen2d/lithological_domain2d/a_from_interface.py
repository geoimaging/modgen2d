# This file is part of geomodgen2D a Python package for ...
# Copyright (C) XXXX Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
# LICENSE

"""Define a two-dimensional domain that defines lithology."""
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
import numpy as np
import copy,warnings

from geomodgen2d.discretized_domain2d import DiscretizedDomain2D
from geomodgen2d.discretized_interfaces2d import DiscretizedInterfaces2D
from geomodgen2d.global_soil_interface_config import GlobalSoilInterfaceConfig

from .a_base import LithologicalDomain2DReadOnly
from .a_from_obs2d import LithologicalDomain2DFromObstruction2D
from .common_functions import _warn_if_changed, _merge_lithological_domains

class LithologicalDomain2D(LithologicalDomain2DReadOnly):
    def __init__(self, domain:DiscretizedDomain2D=None, gwt_depth=None, name:str = ''): 
        """
        Generates a LithogolicalDomain3D instance from the provided GlobalSoilInterfaceConfig.

        Parameters:
        ----------
        domain:
            Discretized Domain of the lithological domain
            If None, uses same domain as GlobalSoilInterfaceConfig's merged interface.
        gwt_depth:
            GWT Depth in unit length. If None, assumed at the bottom/inf.
        name: str
            The name of lithologicaldomain
        """
        self.name = name
        discretizedInterfaces2D_instance = GlobalSoilInterfaceConfig.get_interface_instance()
        
        if domain is not None:
            if not discretizedInterfaces2D_instance.domain.is_equivalent(domain):
                raise ValueError(f"Spans of interfaces [{discretizedInterfaces2D_instance.domain.spans}] and domain provided, [{domain.spans}], are not same.")
        
            discretizedInterfaces2D_instance = discretizedInterfaces2D_instance.remesh_interface(domain.dhs[0], domain.dhs[1])
        
        super().__init__(domain, name)
        
        discretizedInterfaces2D_instance = copy.deepcopy(discretizedInterfaces2D_instance)  #Makes sure the change in the object is local to this function only.
        self.gwt_depth = gwt_depth
        self.lm_type = 'from_interface_config'
        self.lithological_matrix = _layer_id_faster(discretizedInterfaces2D_instance)
        self.lithological_matrix = self.lithological_matrix.astype(int)
        self.lithological_matrix = np.vectorize(lambda x: f"{x}")(self.lithological_matrix)
        self.obstruction_overlap = False #Overlap with merged layers (useful for utils)
        
        self.interface_config_revision_id = GlobalSoilInterfaceConfig.get_revision_id()
        
    @LithologicalDomain2DReadOnly.lithological_matrix.setter
    def lithological_matrix(self, value):
        if value is not None:
            if self.check_for_Xs(value):
                raise ValueError(
                    "LithologicalDomain2D cannot contain 'X' values."
                )

        super(LithologicalDomain2D, type(self)).lithological_matrix.fset(self, value)
        
    def refresh(self):
        """
        Recompute the lithological domain using the latest global interface config.

        Notes
        -----
        - This is called automatically if the interface configuration revision ID changes.
        """
        if self.merged_lit:
            warnings.warn("The current lit domain was created with merging a domain with soil lit domain. That merging domain is lost/ignored on returned lit domain.")
        # Create a fresh instance of same class
        
        init_lithological_matrix = self.lithological_matrix
        domain = self.domain if self.init_domain is None else self.init_domain
        self.__init__(domain, self.gwt_depth, self.name)
        
        _warn_if_changed(self.lithological_matrix, init_lithological_matrix)
        
    def return_merged_lithological_domain(self, lithological_domain2D_list=[]):
        """
        Merge this LithologicalDomain2D with a list of obstruction-based lithological domains.

        Parameters
        ----------
        lithological_domain2D_list : list of LithologicalDomain2DFromObstruction2D
            List of obstruction-based lithological domains to merge into this one.

        Returns
        -------
        merged_lit_domain : LithologicalDomain2D
            A new instance of LithologicalDomain2D representing the merged result.
        overlap_map : np.ndarray
            Boolean matrix indicating where obstructions overlapped with the lithology.
            (Returned from merge_lithological_domains)

        Notes
        -----
        - If the global interface configuration has changed since creation, the domain
          is refreshed before merging.
        - Only obstruction-based lithological domains should be provided in the list.
        """
        if not GlobalSoilInterfaceConfig.get_config_status(self.interface_config_revision_id):
            self.refresh() #Compute for new surface
        
        if self.merged_lit:
            warnings.warn("The current lit domain was created with merging a domain with soil lit domain. That merging domain is lost/ignored on returned lit domain.")
        # Create a fresh instance of same class
        
        domain = self.domain if self.init_domain is None else self.init_domain
            
        merged_lit_domain = self.__class__(domain, self.gwt_depth, self.name)
        
        domains = []
        for lit_domain in lithological_domain2D_list:
            if not isinstance(lit_domain, LithologicalDomain2DFromObstruction2D):
                raise TypeError(
                    "Entries of lithological_domain2D_list must be instances of "
                    "LithologicalDomain2DFromObstruction2D."
                )
            domains.append(lit_domain.domain)
        
        min_domain = DiscretizedDomain2D.get_minimum_domain(domains)
        
        merged_lit_domain.remeshing_lithological_matrix(*min_domain.dhs)
        
        for lit_domain in lithological_domain2D_list:
            if not GlobalSoilInterfaceConfig.get_config_status(lit_domain.interface_config_revision_id):
                lit_domain.refresh() #Compute for new surface
        
            lit_domain = lit_domain.remeshing_lithological_matrix(*min_domain.dhs, replace=False)
            merged_lit_domain.lithological_matrix, _ = _merge_lithological_domains(
                merged_lit_domain, lit_domain
            )
        
        merged_lit_domain.merged_lit = True
        return merged_lit_domain
    
    @classmethod
    def from_config(cls, config_dict):
        if not isinstance(config_dict, dict):
            raise TypeError("Expected a dictionary.")
        
        INTERFACE_CLASSES = (LithologicalDomain2D, LithologicalDomain2DReadOnly)
        
        lm_type = config_dict['lm_type']
        if lm_type.startswith("from_interface_config"):
            if cls not in INTERFACE_CLASSES:
                raise TypeError(
                    f"{cls.__name__}.from_config cannot load lm_type='{lm_type}'. "
                    "Use LithologicalDomain2D.from_config or LithologicalDomain2DReadOnly.from_config instead."
                )

        elif lm_type.startswith("from_obs2D"):
            if cls in INTERFACE_CLASSES:
                raise TypeError(
                    f"{cls.__name__}.from_config cannot load lm_type='{lm_type}'. "
                    "Use LithologicalDomain2DFromObstruction2D.from_config or LithologicalDomain2DReadOnly.from_config instead."
                )

        else:
            raise ValueError(f"Unknown lm_type prefix: '{lm_type}'.")
        
        return super().from_config(config_dict)
        
def _layer_id_faster(discretizedInterfaces2D_instance:DiscretizedInterfaces2D):
    """
    Assigns layer IDs to soil points based on processed boundary values and spatial depth ranges.

    Args:
        processed_boundary (np.ndarray): 
            Boundary values for each layer (z,x format).
        n_layers (int): 
            Number of layers.
        spatial_z_ranges (np.ndarray): 
            Depth values defining layer boundaries.

    Returns:
        np.ndarray: Matrix containing assigned layer IDs.
    """
    # Will be layered 1,2,3...
    # Initializing all soil points are last layers
    n_layers = discretizedInterfaces2D_instance.n_soil_layers
    
    # Ignore the two edges of interfaces_matrix; they serve only for remeshing and fall outside the model bounds.
    processed_boundary = discretizedInterfaces2D_instance.interfaces_matrix
    processed_boundary = processed_boundary[1:-1, :]

    spatial_z_ranges = discretizedInterfaces2D_instance.domain.z_centers
    
    layer_matrix = np.full((processed_boundary.shape[0], len(spatial_z_ranges)), n_layers)

    compare_matrix = np.ones_like(layer_matrix)*spatial_z_ranges.T[None]
    # print(compare_matrix)
    processed_boundary[processed_boundary <= 0] = -1
    processed_boundary[processed_boundary >= spatial_z_ranges[-1]] = spatial_z_ranges[-1]+1
    
    for i in range(n_layers):
        boundary_matrix = np.tile(processed_boundary[:,i], (len(spatial_z_ranges), 1)).T
        # print(boundary_matrix)
        layer_matrix-=(boundary_matrix>=compare_matrix)

    return layer_matrix
