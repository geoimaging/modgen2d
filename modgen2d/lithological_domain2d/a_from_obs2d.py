# This file is part of modgen2d a Python package for ...
# Copyright (C) XXXX Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
# LICENSE

"""Define a 2D lithological domain derived from obstruction (utility) data."""
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
import numpy as np
import modgen2d.general_functions as f
import copy

from modgen2d.discretized_domain2d import DiscretizedDomain2D
from modgen2d.obstruction2d import Obstruction2D
from modgen2d.interface.global_soil_interface_config import GlobalSoilInterfaceConfig
from .a_base import LithologicalDomain2DReadOnly
from .common_functions import _warn_if_changed, _merge_lithological_domains
  
class LithologicalDomain2DFromObstruction2D(LithologicalDomain2DReadOnly):
    """
    Represents a 2D lithological domain generated from obstruction data (utilities, pipes, etc.).
    This class allows merging multiple 2D obstruction datasets into a lithological matrix.
    """
    def __init__(self, domain:DiscretizedDomain2D, name: str=''):
        """
        Initialize a 'LithologicalDomain2DFromObstruction2D' instance.

        Parameters
        ----------
        domain : DiscretizedDomain2D
            Discretized spatial domain for the lithological matrix.
        name : str, optional
            Name of the lithological domain.
        """
        super().__init__(domain, name)
        self.domain = domain
        self.name = name

        assert self.lm_type == 'NA', f"ERROR: The variable is already assigned with {self.lm_type}, should be 'NA'."
        self.obstruction2d_dict_list = []        
        self.obstruction_overlap = False
        
    def add_obstruction2D(self, obstruction2D_instance:Obstruction2D, shift_ref2d_to_xy, feature_id):
        """
        Add a 2D obstruction dataset to the lithological domain.

        Parameters
        ----------
        obstruction2D_instance : Obstruction2D
            Obstruction dataset (e.g., utilities) to incorporate.
        shift_ref2d_to_xy : array-like of shape (2,)
            Coordinates [x, y] to shift the reference point of the obstruction data.
        feature_id : str, optional
            Prefix for labeling this obstruction in the lithological matrix.
            Must be <=8 characters, no underscores or numbers. Use 'def' for soil.

        Raises
        ------
        ValueError
            If shift_ref2d_to_xy shape is invalid or feature_id is not valid.
        AssertionError
            If obstruction2D_instance is improperly defined.
        """
        ## Do all checks.
        if not GlobalSoilInterfaceConfig.get_config_status(self.interface_config_revision_id):
            self.refresh() #Compute for new surface
        
        shift_ref2d_to_xy = np.asarray(shift_ref2d_to_xy)
        if shift_ref2d_to_xy.shape != (2,):
            raise ValueError("shift_ref2d_to_xy must have shape (2,)")

        valid_prefix, msg = f.is_valid_feature_id(feature_id)
        if not valid_prefix:
            raise ValueError(msg)
        
        # Note utils_3d is already shifted
        obstruction2D_instance = copy.deepcopy(obstruction2D_instance)  #Makes sure the change in the object is local to this function only.
        assert obstruction2D_instance.shape is True, "obstruction2D_instance is not defined properly"
                
        # Get y_shift_from_surfaceBoundaryConfig
        y_shift_for_surface_adj = self._get_y_shift_adjusted_for_surface(shift_ref2d_to_xy)
        shift_ref2d_to_xy[1]+=y_shift_for_surface_adj
        
        # Get Lithological Matrix from obstacle
        X, Z = np.meshgrid(self.domain.x_centers, self.domain.z_centers, indexing='ij')
        points_orig = np.vstack([X.ravel(), Z.ravel()]).T  # Shape: (M, 3)
        N = points_orig.shape[0]
        
        chunk_size = 1000000 #Max chuck per one time.
        vals_all = np.zeros(N, dtype=int)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            vals_all[start:end] = obstruction2D_instance.query_points_in_obstruction(points_orig, shift_ref2d_to_xy)    
        vals_all = vals_all.reshape(X.shape)
        lithological_domain_matrix = vals_all.astype(int)

        expected_ids = np.unique(obstruction2D_instance.grid2d)
        expected_ids = expected_ids[expected_ids != 0]

        if feature_id != 'def':
            lithological_domain_matrix = np.vectorize(lambda x: f"{feature_id}_{x}")(lithological_domain_matrix)
            mask_b_nonzero = (lithological_domain_matrix == f"{feature_id}_0")
            lithological_domain_matrix = np.where(mask_b_nonzero, 'X', lithological_domain_matrix)  
            expected_ids = [f"{feature_id}_{str(x)}" for x in expected_ids]
        else:
            # 'def' means for soil. Note '0' is not allowed.
            lithological_domain_matrix = np.vectorize(lambda x: f"{x}")(lithological_domain_matrix)
            mask_b_nonzero = (lithological_domain_matrix == f"0")
            lithological_domain_matrix = np.where(mask_b_nonzero, 'X', lithological_domain_matrix)  
            expected_ids = [f"{str(x)}" for x in expected_ids]
        
        expected_ids.append("X")
        self._set_lit_ids_expected(expected_ids, merge=True)
        
        obstruction2D_dict = {
            'obstruction_inst': obstruction2D_instance,
            'shift_ref2d_to_xy': shift_ref2d_to_xy,        
            'feature_id': feature_id,   
            'y_shift_for_surface_adj': y_shift_for_surface_adj,
        }  
        new_lit_domain_dict = {
            'lm_type': 'from_obs2D',
            'domain':self.domain,
            'lithological_matrix':lithological_domain_matrix,
            'interface_config_revision_id':self.interface_config_revision_id,
        }
        
        ## Merging with all existing lithological matrix if exist
        self.lithological_matrix, obstruction_overlap = _merge_lithological_domains(self, new_lit_domain_dict)        
        self.lm_type = 'from_obs2D'
        self.obstruction_overlap |= obstruction_overlap #Or
        self.obstruction2d_dict_list.append(obstruction2D_dict)  
        
    def refresh(self):
        """
        Refresh the lithological domain and reapply all obstruction datasets.
        """
        init_lithological_matrix = self.lithological_matrix
        
        obstruction2d_dict_list = self.obstruction2d_dict_list
        self.__init__(self.domain, self.name)
        for obs in obstruction2d_dict_list:
            self.add_obstruction2D(obs['obstruction_inst'], obs['shift_ref2d_to_xy'], obs['feature_id'])
        
        _warn_if_changed(self.lithological_matrix, init_lithological_matrix)
        
    def _get_y_shift_adjusted_for_surface(self, shift_ref2d_to_xy):
        ## Check if surface config changed after definition: All this is handled by redefining this class (in merged case.)
        _, surface_interface = GlobalSoilInterfaceConfig.get_interface_instance().get_surface_and_subsurface_interfaces()
        
        ## Make sure domains dhs are consistent
        if not (
            len(self.domain.spans) == len(surface_interface.domain.spans)
            and all(f.is_close(a, b) for a, b in zip(self.domain.spans,
                                                surface_interface.domain.spans))
        ):
            raise ValueError(
                "The domains' spans are not consistent. "
                f"Lithological domain has spans {self.domain.spans}, "
                f"while surface interface has {surface_interface.domain.spans}"
            )
    
        if all(f.is_close(a, b) for a, b in zip(self.domain.spans,
                                                surface_interface.domain.spans)):
            surface_interface.remesh_interface(self.domain.dhs[0], self.domain.dhs[1])
    
        # Find coord in surface_interface nearest to x_coord of .
        surface_x_coords = surface_interface.domain.get_interface_x_centers
        interp_ref_zs = f.remeshing_2D_matrix(x_old = surface_x_coords, x_new = [shift_ref2d_to_xy],
                                    z_old = [1], z_new = [1], matrix_2d = surface_interface.interfaces_matrix, interp_method = surface_interface.remesh_interp_method)


        # Find the x_coordinate
        z_shift = interp_ref_zs[0]    
        return z_shift

    @classmethod
    def from_config(cls, config_dict):
        """
        Create a 'LithologicalDomain2DFromObstruction2D' instance from a configuration dictionary.
        """
        if not isinstance(config_dict, dict):
            raise TypeError("Expected a dictionary.")
        
        OBS_CLASSES       = (LithologicalDomain2DFromObstruction2D, LithologicalDomain2DReadOnly)
        
        lm_type = config_dict['lm_type']
        if lm_type.startswith("from_interface_config"):
            if cls in OBS_CLASSES:
                raise TypeError(
                    f"{cls.__name__}.from_config cannot load lm_type='{lm_type}'. "
                    "Use LithologicalDomain2D.from_config or LithologicalDomain2DReadOnly.from_config instead."
                )

        elif lm_type.startswith("from_obs2D"):
            if cls not in OBS_CLASSES:
                raise TypeError(
                    f"{cls.__name__}.from_config cannot load lm_type='{lm_type}'. "
                    "Use LithologicalDomain2DFromObstruction2D.from_config or LithologicalDomain2DReadOnly.from_config instead."
                )

        else:
            raise ValueError(f"Unknown lm_type prefix: '{lm_type}'.")
        
        return super().from_config(config_dict)
        
        