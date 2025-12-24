# This file is part of geomodgen2D a Python package for ...
# Copyright (C) XXXX Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
# LICENSE

"""Define a two-dimensional domain that defines lithology."""
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
import numpy as np
import matplotlib.pyplot as plt
import geomodgen2d.general_functions as f
import copy,scipy,warnings
import matplotlib.colors as mcolors

from geomodgen2d.discretized_domain2d import DiscretizedDomain2D
from geomodgen2d.interfaces_creator2d import AbstractInterfacesCreator2D
from geomodgen2d.discretized_interfaces2d import DiscretizedInterfaces2D, SurfaceInterface2D
from geomodgen2d.obstruction2d import Obstruction2D
from geomodgen2d.meta_class import _StrictProtectedMeta, _internal_classmethod

class GlobalSoilInterfaceConfig(metaclass=_StrictProtectedMeta):
    """
    Global configuration manager for the active surface-interface instance and 
    its processing behavior. This class acts as a centralized registry that 
    stores the current interface, processing mode, and a revision token used 
    to detect boundary changes.
    """

    __slots__ = []  #Does not allow any instance variables. Only class variables.
    
    # central definition of DEFAULTS
    _DEFAULTS = {
        "_soil_interface2d_instance": None,
        "_surface_interface2d_instance": None,
        "_merged_interface2d_instance": None,
        "_surface_interface_method": None,
        "_revision_id": 0,
        "_status_code": 0,
    }
    
    # initialize class attributes once
    # default values
    _soil_interface2d_instance = None
    _surface_interface2d_instance = None
    _merged_interface2d_instance = None
    _surface_interface_method = None
    _revision_id = 0
    _status_code = 0

        
    @_internal_classmethod
    def reset(cls):
        for key, val in cls._DEFAULTS.items():
            setattr(cls, key, val)
    
    @_internal_classmethod
    def set_soil_interface(cls, soil_interface2d_instance:DiscretizedInterfaces2D, 
                           surface_interface2d_instance:AbstractInterfacesCreator2D = None,
                           surface_interface_method="pile", 
                            #   compute_immediately=False, 
                           force_set=False):
        """
        Set the global soil interface configuration.

        Parameters
        ----------
        soil_interface_class : DiscretizedInterfaces2D
            The soil interface instance to activate globally.
        
        surface_interface2d_instance : DiscretizedInterfaces2D
            The surface interface instance to activate globally.
            surface_included_in_soil_interface must be False.
        
        surface_interface_method : str, default="pile"
            Surface modification method. Valid options:
            - "pile"  : add material
            - "erode" : remove material

        force_set : bool, default=False
            If False and a surface interface is already set, a RuntimeError 
            will be raised. If True, the existing interface is overwritten.
            
        Generates:
        _revision_id : int or None
            A randomly generated integer that uniquely identifies the 
            current configuration state. Updated every time 
            `set_surface_interface()` is called. Allows downstream systems to 
            detect changes in surface boundaries or processing modes.
            
        _status_code: int
            Either 0, 1, 2, or 99. with 0 being the best, and 99 being the worst.
        """
        soil_interface2d_instance._locked = False
        
        if cls.get_revision_id() != 0 and not force_set:
            raise RuntimeError(
                "Surface interface already set. "
                "Use force-set=True if you intentionally want to overwrite it.")
        
        if soil_interface2d_instance is None:
            raise TypeError("soil_interface2d_instance cannot be None")
        
        if not isinstance(soil_interface2d_instance, AbstractInterfacesCreator2D):
            raise TypeError("soil_interface2d_instance must be from subclass of AbstractInterfacesCreator2D class.")
        
        if isinstance(soil_interface2d_instance, SurfaceInterface2D):
            raise TypeError("soil_interface2d_instance cannot of SurfaceInterface2D class.")

        soil_interface2d_instance.lock_interfaces()
        remesh = soil_interface2d_instance.remesh_interp_method
        if surface_interface2d_instance is not None:
            surface_interface2d_instance._locked = False
            if surface_interface2d_instance.n_interfaces != 1:
                raise ValueError(f"interface_class must have exactly one interface. Provided {surface_interface2d_instance.n_interfaces}")
            if surface_interface2d_instance.remesh_interp_method != remesh:
                raise ValueError(f"Remesh interp method for both surface and soil interfaces must be same. Provided {surface_interface2d_instance.remesh_interp_method} and {remesh} respectively.")
            
            surface_interface2d_instance._adjust_for_top_surface_interface()
            surface_interface2d_instance.lock_interfaces()

        if surface_interface_method not in ['pile', 'erode']:
            raise ValueError(f"Methods can only be either 'pile' or 'erode'. Provided: {surface_interface_method}")
        
        cls._soil_interface2d_instance = soil_interface2d_instance
        cls._surface_interface2d_instance = surface_interface2d_instance
        cls._merged_interface2d_instance = soil_interface2d_instance.get_interfaces_matrix_with_surface(surface_interface2d_instance, surface_interface_method)
        cls._surface_interface_method = surface_interface_method
        
        low = 1
        high = 2**63    # exclusive upper bound
        magnitude = np.random.randint(low, high, dtype=np.int64)
        sign = 1 if np.random.random() < 0.5 else -1
        unique_code = magnitude * sign
        cls._revision_id  = int(unique_code) 
            
    @_internal_classmethod
    def get_revision_id(cls):
        return cls._revision_id
    
    @_internal_classmethod
    def get_interface_instance(cls, type=None):
        if type is None or type == 'merged':
            return cls._merged_interface2d_instance
        elif type == 'surface':
            return cls._surface_interface2d_instance
        elif type == 'soil_only':
            return cls._soil_interface2d_instance
        else:
            raise ValueError(f"Types can be either 'surface', 'soil_only', or 'merged'/None. Provided {type}.")

    @_internal_classmethod    
    def get_config_status(cls, previous_revision_id):
        """
        Check whether an external module's cached configuration is still valid.
        
        Parameters
        ----------
        previous_revision_id : int
            The revision ID previously recorded by the caller.
        
        Returns
        -------
        Boolean
            True  : Fully consistent — same revision, same compute mode.
            False : Revision changed
        """
        current_revision_id = cls.get_revision_id()
        return previous_revision_id == current_revision_id
    
class LithologicalDomain2DReadOnly():
    """
    Class representing a 2D matrix (with layer_ID) with layers that can be created from a boundary or utility class.
    """
    def __init__(self, domain:DiscretizedDomain2D, name: str):
        """
        Initializes the LithogolicalDomain2D instance with given spatial limits, and spacing.
        
        Parameters:
        domain: DiscretizedDomain3D
            Domain in which the interfaces are to be defined.
        name: str
            The name of lithologicaldomain
        """
        self.domain = domain
        self.name = name
        self.lm_type = 'NA'
        self.read_only=False
        self.lithological_matrix = None
        self.interface_config_revision_id = GlobalSoilInterfaceConfig.get_revision_id()
        
        #For lithologicalDomain from Interface
        self.gwt_depth = None
        
        #For Lithological Domain from Obstruction2D
        self.obstruction2D_dict = None

        self.merged_lit = False
        self.init_domain = None #None means domain has never been changed.
        
        # lit_order to be used as order while simulating.
        self.lit_order = None
        
    def print(self):
        print(f"N_x_coord = {self.lithological_matrix.shape[0]}, N_z_coord = {self.lithological_matrix.shape[1]}")
        print("Layered Matrix : \n", self.lithological_matrix.T) 
    
    ##TODO add unittests
    def get_feature_id_and_lit_val_from_lithological_matrix(self):
        lithological_matrix = self.lithological_matrix

        if lithological_matrix is None:
            return {}

        if not isinstance(lithological_matrix, np.ndarray):
            raise TypeError(
                "lithological_matrix must be None or a numpy array."
            )

        # Use only unique entries (much faster and avoids duplicates)
        arr = np.unique(lithological_matrix.astype(str))
        arr = arr[arr != 'X']
        
        # mask for pure digits
        mask_def = np.char.isdigit(arr)

        # Everything else is candidate for prefix-number
        non_def = arr[~mask_def]

        # Partition by underscore
        sep = '_'
        result = {"def": arr[mask_def].astype(int).tolist()}
        for s in non_def:
            parts = s.split(sep, 1)

            # Case 1: no underscore → prefix only
            if len(parts) == 1:
                prefix = parts[0]
                suffix = None
            else:
                prefix, suffix = parts

                # ERROR: suffix must be numeric
                if not suffix.isdigit():
                    raise ValueError(
                        f"Invalid numeric suffix in: {s}"
                    )

                suffix = int(suffix)

            result.setdefault(prefix, [])
            if suffix is not None:
                result[prefix].append(suffix)
        return result
    
    def check_shape(self):
        domain_shape = self.domain.shape
        lit_shape = self.lithological_matrix.shape
        if domain_shape != lit_shape:
            raise ValueError(f"Matrix shape mismatch. Domain shape {domain_shape} != lit_shape {lit_shape}.")
        
    @staticmethod
    def __get_unique_lithological_color_map(
        lithological_matrix,
        color_map = {
        'def': plt.get_cmap('tab20', 10),      # For integer values
        'U_': plt.get_cmap('Set3', 10)   # For "U-{x}" values
    }):
        unique_values = np.unique(lithological_matrix)
        color_mapping = {} 
        for value in unique_values:
            assigned = False
            # print(value, f.is_integer_value(value))
            for prefix, cmap in color_map.items():
                if value == 'X':
                    color_mapping[value] = (1.,1.,1.,1.)#'#ffffff'
                    assigned = True
                    
                elif prefix == 'def' and f.is_integer_value(value):
                    if value == 0 or value == '0':
                        color_mapping[value] = (1.,1.,1.,1.)#'#ffffff'
                    else:
                        # If no prefix, assume integer or digit
                        index = int(float(value)) % 10
                        color_mapping[value] = cmap(index)
                    assigned = True
                    break
                elif isinstance(value, str) and value.startswith(prefix):
                    # For prefixed values
                    index = int(float(value[len(prefix):])) % 10
                    color_mapping[value] = cmap(index)
                    assigned = True
                    break
            
            # If no pattern matched, assign a random color
            if not assigned:
                color_val = "#" + ''.join([np.random.choice(list('0123456789ABCDEF')) for _ in range(6)])
                color_mapping[value] = mcolors.to_rgba(color_val)
                
        # Create a colormap from the color mapping
        colors = [color_mapping[value] for value in unique_values]
        int_map = {value: color_mapping[value] for idx, value in enumerate(unique_values)}
        integer_mapped_array = np.array(np.vectorize(int_map.get)(lithological_matrix), dtype='float')
        integer_mapped_array = np.transpose(integer_mapped_array, axes=(2, 1, 0))  #Adjusting for imshow
        fixed_cmap = mcolors.ListedColormap(colors)

        return unique_values, color_mapping, integer_mapped_array, fixed_cmap
    
    def plot(self, ax=None, discrete_point_size=0, legend=True,
               id2material_dict = None, title='Lithological Domain',
               plot_interfaces = False,
               color_map = {
                        'def': plt.get_cmap('tab20', 10),      # For integer values
                        'U_': plt.get_cmap('Set3', 10)   # For "U-{x}" values
                }):
        """
        # If any change change in materialdomain too.
        
        Plots a 2D section of the layered matrix.

        Parameters:
            ax: The matplotlib axes object for the plot (default is None, which creates a new figure).
            idx: A list specifying the axis and index to slice (e.g., ['x', 0]).
            color_map: A dictionary that defines the color map for the values in the matrix.
        """
        if ax is None:
            fig,ax = plt.subplots()

        z_centers, x_centers = self.domain.z_centers, self.domain.x_centers
        span_x, span_z = self.domain.spans
        
        unique_values, color_mapping, integer_mapped_array, fixed_cmap = LithologicalDomain2DReadOnly.__get_unique_lithological_color_map(self.lithological_matrix, color_map)
        
        # Plot the data using imshow with the fixed colormap
        extent = [0, span_x, span_z, 0]
        cax = ax.imshow(integer_mapped_array, cmap=fixed_cmap, extent=extent, interpolation='none')
        
        # Plot gwt
        if self.gwt_depth is not None:
            edges_kw = dict(color='r', linestyle='dashed', linewidth=2, zorder=4000)
            ax.plot([0, span_x], [self.gwt_depth, self.gwt_depth], **edges_kw)


        x_data, z_data = np.meshgrid(x_centers, z_centers, indexing='ij')
        if discrete_point_size!=0:
            ax.scatter(x_data.flatten(), z_data.flatten(), c = [color_mapping[value] for value in self.lithological_matrix.flatten()], edgecolors='k', s=discrete_point_size)
            
        # Plot Boundary:
        if plot_interfaces:
            discretizedInterfaces2D_instance = GlobalSoilInterfaceConfig.get_interface_instance()
            if discretizedInterfaces2D_instance is not None:
                n_interface = discretizedInterfaces2D_instance.n_interfaces
                for i in np.arange(n_interface-1, -1, -1):
                    remesh_tech = discretizedInterfaces2D_instance.remesh_interp_method
                    if remesh_tech == 'nearest':
                        drawstyle = 'steps-mid'
                    elif remesh_tech == 'linear':
                        drawstyle = 'default'
                        if self.init_domain is not None:
                            warnings.warn("Looks like the lit domain has been remeshed with linear after creation.")
                    else:
                        warnings.warn(f"Interfaces might not reflect the exact interpolation in the plots except for 'linear' and 'nearest'. Provided {remesh_tech}.")
                    ax.plot(discretizedInterfaces2D_instance.domain.get_interface_x_centers,
                            discretizedInterfaces2D_instance.interfaces_matrix[:, i],
                            linestyle='--',
                            color='k',
                            drawstyle=drawstyle,
                        )
        
        # Create a custom legend
        handles = [plt.Line2D([0], [0], marker='s', color=color_mapping[value], markersize=10, linestyle='') for value in unique_values]
        gwt_handle = plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='GWT')
        handles.append(gwt_handle)
        labels = list(unique_values) + ['GWT']
    
        if id2material_dict is not None:
            labels = [id2material_dict[label][1] if label in id2material_dict else label for label in labels]
            labels = [lbl.decode('utf-8') if isinstance(lbl, bytes) else lbl for lbl in labels]
        
        if legend:
            ax.legend(handles, unique_values, title="Legend", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        if title is not None:
            ax.set_title(title)
            
        ax.axis('scaled')
        ax.set(
            xlim= [0, span_x],
            ylim= [span_z, 0],
            xlabel='X',
            ylabel='Z',
        )
        return ax

    def remeshing_lithological_matrix(self, new_dx, new_dz, interp_method = 'nearest', replace=True):
        """
        Coarsens/refines the layered matrix based on a coarsened coordinate class, optionally replacing the current matrix.

        Args:
            del_x_remeshed, del_z_remeshed: Spacing of remeshed (new) del_x and del_z.
            interp_method (str): The interpolation method to use ('nearest' or 'nearest_up'). Defaults to 'nearest'.
            replace (bool): Whether to replace the current matrix with the coarsened one. Defaults to False.
        """
        self_copy = copy.deepcopy(self)  #Makes sure the change in the object is local to this function only.
        org_dx, org_dz = self.domain.dhs
        if org_dx != new_dx or org_dz == new_dz:

            if interp_method!='nearest' and interp_method!='nearest_up':
                raise ValueError(f"interp_method cannot be other than 'nearest' or 'nearest_up' for replacement. Provided {interp_method}")
                
            new_domain = self.domain.remesh(new_dx, new_dz)
            
            unique_values, int_mapp = np.unique(self.lithological_matrix, return_inverse=True)
            int_mapp = int_mapp.reshape(self.lithological_matrix.shape)
            int_values = np.arange(len(unique_values))
            
            remeshed_int_mapp = f.remeshing_2D_matrix(x_old = self.domain.x_centers, 
                                                      z_old = self.domain.z_centers, 
                                                      x_new = new_domain.x_centers,
                                                      z_new = new_domain.z_centers,
                                                      matrix_2d=int_mapp, interp_method=interp_method)
            
            self_copy.domain = new_domain
            self_copy.lithological_matrix = unique_values[remeshed_int_mapp.astype(int)]
            self_copy.lm_type = f'remeshed_{self_copy.lm_type}'
            if self_copy.init_domain is None:
                self_copy.init_domain = self.domain

        if replace:  #to verify
            # Update `self`'s attributes in-place to reflect `self_copy`
            self.__dict__.update(self_copy.__dict__)
        else:
            return self_copy
    
    @property
    def get_config(self):
        pass
    
    @classmethod
    def from_config(cls, config_dict):
        pass
      
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
        self.lithological_matrix = layer_id_faster(discretizedInterfaces2D_instance) - 1 #-1 as top layer is "air" with ID 0, not default 1.
        self.lithological_matrix = self.lithological_matrix.astype(int)
        self.lithological_matrix = np.vectorize(lambda x: f"{x}")(self.lithological_matrix)
        
        self.obstruction_overlap = False #Overlap with merged layers (useful for utils)
        self.obstruction_description = 'Interfaces'
        
        self.interface_config_revision_id = GlobalSoilInterfaceConfig.get_revision_id()
        
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
            merged_lit_domain.lithological_matrix, _ = merge_lithological_domains(
                merged_lit_domain, lit_domain
            )
        
        merged_lit_domain.merged_lit = True

        return merged_lit_domain
        
class LithologicalDomain2DFromObstruction2D(LithologicalDomain2DReadOnly):
    def __init__(self, domain:DiscretizedDomain2D, name: str=''):
        """
        Generates a layered matrix from a 2D utilities class.
        
        Parameters:
        domain: DiscretizedDomain2D
            Domain for this lithological domain.
        obstruction2D_instance: 
            An Obstruction2D object that provides 2D obstruction data.
        shift_2d_ref_to_xy: Format [x,y]
            Reference point of Obstruction2D object is shifted to this shift_2d_ref_to_xy coord.
        added_prefix: Optional
            Optional prefix (Max: 8) to be added to the merged matrix (default is None). Cannot have "_", or numbers, also cannot be "".
        name: str
            The name of lithologicaldomain
        """
        super().__init__(domain, name)
        self.domain = domain
        self.name = name

        assert self.lm_type == 'NA', f"ERROR: The variable is already assigned with {self.lm_type}, should be 'NA'."
        self.obstruction2d_dict_list = []        
        self.obstruction_overlap = False
        
    def add_obstruction2D(self, obstruction2D_instance:Obstruction2D, shift_ref2d_to_xy, added_prefix=None):
        ## Do all checks.
        if not GlobalSoilInterfaceConfig.get_config_status(self.interface_config_revision_id):
            self.refresh() #Compute for new surface
        
        shift_ref2d_to_xy = np.asarray(shift_ref2d_to_xy)
        if shift_ref2d_to_xy.shape != (2,):
            raise ValueError("shift_ref2d_to_xy must have shape (2,)")

        if added_prefix is not None:
            valid_prefix, msg = f.is_valid_feature_id(added_prefix)
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

        if added_prefix is not None:
            lithological_domain_matrix = np.vectorize(lambda x: f"{added_prefix}_{x}")(lithological_domain_matrix)
            mask_b_nonzero = (lithological_domain_matrix == f"{added_prefix}_0")
            lithological_domain_matrix = np.where(mask_b_nonzero, 'X', lithological_domain_matrix)  
            self.added_prefix = True
            
        obstruction2D_dict = {
            'obstruction_inst': obstruction2D_instance,
            'shift_ref2d_to_xy': shift_ref2d_to_xy,        
            'added_prefix': added_prefix,   
            'y_shift_for_surface_adj': y_shift_for_surface_adj,
        }  
        new_lit_domain_dict = {
            'lm_type': 'lith',
            'domain':self.domain,
            'lithological_matrix':lithological_domain_matrix,
            'interface_config_revision_id':self.interface_config_revision_id,
        }
        
        ## Merging with all existing lithological matrix if exist
        self.lithological_matrix, obstruction_overlap = merge_lithological_domains(self, new_lit_domain_dict)        
        self.lm_type = 'lith'
        self.obstruction_overlap |= obstruction_overlap #Or
        self.obstruction2d_dict_list.append(obstruction2D_dict)  
        
    def refresh(self):
        init_lithological_matrix = self.lithological_matrix
        
        obstruction2d_dict_list = self.obstruction2d_dict_list
        self.__init__(self.domain, self.name)
        for obs in obstruction2d_dict_list:
            self.add_obstruction2D(obs['obstruction_inst'], obs['shift_ref2d_to_xy'], obs['added_prefix'])
        
        _warn_if_changed(self.lithological_matrix, init_lithological_matrix)
        
    def _get_y_shift_adjusted_for_surface(self, shift_ref2d_to_xy):
        ## Check if surface config changed after definition: All this is handled by redefining this class (in merged case.)
        _, surface_interface = GlobalSoilInterfaceConfig.get_interface_instance().seperate_surface_interface()
        
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

def layer_id_faster(discretizedInterfaces2D_instance:DiscretizedInterfaces2D):
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
    n_layers = discretizedInterfaces2D_instance.n_layers
    
    # Ignore the two edges of interfaces_matrix; they serve only for remeshing and fall outside the model bounds.
    processed_boundary = discretizedInterfaces2D_instance.interfaces_matrix
    processed_boundary = processed_boundary[1:-1, :]

    spatial_z_ranges = discretizedInterfaces2D_instance.domain.z_centers
    
    layer_matrix = np.full((processed_boundary.shape[0], len(spatial_z_ranges)), n_layers)

    compare_matrix = np.ones_like(layer_matrix)*spatial_z_ranges.T[None]
    # print(compare_matrix)
    processed_boundary[processed_boundary <= 0] = -1
    processed_boundary[processed_boundary >= spatial_z_ranges[-1]] = spatial_z_ranges[-1]+1
    
    for i in range(n_layers-1):
        boundary_matrix = np.tile(processed_boundary[:,i], (len(spatial_z_ranges), 1)).T
        # print(boundary_matrix)
        layer_matrix-=(boundary_matrix>=compare_matrix)

    return layer_matrix

def _extract_lithological_fields(ld):
    """
    Extract lithological domain fields from either a ReadOnly object or a dict.

    Parameters
    ----------
    ld : LithologicalDomain2DReadOnly or dict
        Either a domain object with attributes (lm_type, domain,
        lithological_matrix, interface_config_revision_id)
        or a dict containing these keys.

    Returns
    -------
    tuple
        A 4-tuple: (lm_type, domain, lithological_matrix, interface_config_revision_id)

    Raises
    ------
    KeyError
        If `ld` is a dict and required keys are missing.
    """
    if isinstance(ld, dict):
        return (
            ld["lm_type"],
            ld["domain"],
            ld["lithological_matrix"],
            ld["interface_config_revision_id"]
        )
    else:
        return (
            ld.lm_type,
            ld.domain,
            ld.lithological_matrix,
            ld.interface_config_revision_id
        )
        
def merge_lithological_domains(lithological_domain_A, lithological_domain_B, lit_id_none_list=['X']):
    """
    Merge two lithological domains (either ReadOnly objects or dicts).

    Priority rule: Values from Domain B overwrite values from Domain A,
    except when B contains a 'none' lithological ID (specified in `lit_id_none_list`).

    Parameters
    ----------
    lithological_domain_A : LithologicalDomain2DReadOnly or dict
        First lithological domain. May be a ReadOnly object or a dict with keys:
        'lm_type', 'domain', 'lithological_matrix', 'interface_config_revision_id'.

    lithological_domain_B : LithologicalDomain2DReadOnly or dict
        Second lithological domain. Priority is given to this domain during merge.
        Must match domain size, extents, and surface revision ID of A.

    lit_id_none_list : list of str, optional
        List of lithology IDs that represent "none" (i.e., no material).
        Defaults to ['X'].

    Returns
    -------
    merged_matrix : np.ndarray
        A numpy array representing the merged lithological matrix.

    overlap : bool
        True if both domains had non-none values at the same grid locations,
        indicating overlapping materials (useful for obstruction/utility detection).

    Notes
    -----
    - Supports both dict-based and ReadOnly object inputs.
    - Domain B overwrites A, unless its value is a 'none' ID.
    - Overlap detection checks where both A and B have valid (non-none) values.
    """

    lm_type_A, domain_A, matrixA, rev_id_A = _extract_lithological_fields(lithological_domain_A)
    lm_type_B, domain_B, matrixB, rev_id_B = _extract_lithological_fields(lithological_domain_B)

    # Checks
    assert lm_type_B != "NA", "lm_type cannot be NA"
    
    if lm_type_A == "NA":
        return matrixB, False

    if domain_A != domain_B:
        raise TypeError(
            f"Domains of A and B do not match. "
            f"A: spans={domain_A.spans}, dhs={domain_A.dhs}; "
            f"B: spans={domain_B.spans}, dhs={domain_B.dhs}"
        )

    if matrixA.shape != matrixB.shape:
        raise ValueError(
            f"Expected same lithological domain matrices size, got {matrixA.shape} != {matrixB.shape}"
        )

    if rev_id_A != rev_id_B:
        raise ValueError("Surface config revision IDs do not match. Both A and B must be for same interfaces configuration. Refresh if needed") ##TODO

    if not isinstance(lit_id_none_list, list) or not lit_id_none_list or not all(isinstance(item, str) for item in lit_id_none_list):
        raise ValueError("lit_id_none_list must be a non-empty list of strings.")

    # Masks to identify non-'none' values
    mask_A_non_none = ~np.isin(matrixA, lit_id_none_list)
    mask_B_non_none = ~np.isin(matrixB, lit_id_none_list)

    # B overwrites A 
    merged_lithological_matrix = np.where(mask_B_non_none, matrixB, matrixA)
    # (replace it with that of B even if merged already have values, i.e prioritize B)

    # Overlap detection - #Check if overlap with merged layers (useful for obstructions)
    overlap = np.sum(mask_A_non_none * mask_B_non_none)>0
    return merged_lithological_matrix, overlap

    # self.lithological_matrix = merged
    # self.overlap = overlap_check>0 #Overlap with merged layers (useful for utils)
    # self.utils_description = f'{self.utils_description} + {lithological_domain_B.utils_description}'

def _warn_if_changed(a, b, msg="On refreshing the lithological domain, lithological_matrix changed."):
    if a is None and b is None:
        changed = False
    elif a is None or b is None:
        changed = True
    elif np.isscalar(a) and np.isscalar(b):
        changed = a != b
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        changed = not np.array_equal(a, b)
    else:
        # one is scalar, the other is array → consider changed
        changed = True

    if changed:
        warnings.warn(msg)




