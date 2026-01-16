# This file is part of geomodgen2D a Python package for ...
# Copyright (C) XXXX Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
# LICENSE

"""Define a two-dimensional domain that defines lithology."""
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
import numpy as np
import matplotlib.pyplot as plt
import geomodgen2d.general_functions as f
import copy,warnings
import matplotlib.colors as mcolors

from geomodgen2d.discretized_domain2d import DiscretizedDomain2D
from geomodgen2d.obstruction2d import Obstruction2D
from geomodgen2d.global_soil_interface_config import GlobalSoilInterfaceConfig

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
        self.lithological_matrix = None
        self.interface_config_revision_id = GlobalSoilInterfaceConfig.get_revision_id()
        
        #For lithologicalDomain from Interface
        self.gwt_depth = None
        
        #For Lithological Domain from Obstruction2D
        self.obstruction2d_dict_list = []        
        self.obstruction_overlap = None
        
        self.merged_lit = False
        self.init_domain = None #None means domain has never been changed.
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
                n_soil_layers = discretizedInterfaces2D_instance.n_soil_layers
                for i in np.arange(n_soil_layers-1, -1, -1):
                    remesh_tech = discretizedInterfaces2D_instance.remesh_interp_method
                    if remesh_tech == 'nearest':
                        drawstyle = 'steps-mid'
                    elif remesh_tech == 'linear':
                        drawstyle = 'default'
                        if self.init_domain is not None:
                            warnings.warn("Looks like the lit domain has been remeshed with linear after creation.")
                    else:
                        drawstyle = 'steps-mid'
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
            self_copy.lm_type = f'{self_copy.lm_type}_remeshed'
            if self_copy.init_domain is None:
                self_copy.init_domain = self.domain

        if replace:  #to verify
            # Update `self`'s attributes in-place to reflect `self_copy`
            self.__dict__.update(self_copy.__dict__)
        else:
            return self_copy
    
    @property
    def get_config(self):
        self_config = {}
        self_config['domain'] = self.domain.get_config
        self_config['gwt_depth'] = self.gwt_depth
        if self.init_domain is None:
            self_config['init_domain'] = self.init_domain
        else:
            self_config['init_domain'] = self.init_domain.get_config
        self_config['interface_config_revision_id'] = self.interface_config_revision_id
        
        self_config['lithological_matrix'] = self.lithological_matrix
        self_config['lm_type'] = self.lm_type
        self_config['merged_lit'] = self.merged_lit
        self_config['name'] = self.name
        self_config['lit_order'] = self.lit_order
       
        self_config['obstruction2d_dict_list'] = {}
        if len(self.obstruction2d_dict_list)!=0:
            for i,obstruction2D_dict in enumerate(self.obstruction2d_dict_list):
                self_config['obstruction2d_dict_list'][f'{i}'] = {
                    'obstruction_inst': obstruction2D_dict['obstruction_inst'].get_config,
                    'shift_ref2d_to_xy': obstruction2D_dict['shift_ref2d_to_xy'],        
                    'added_prefix': obstruction2D_dict['added_prefix'],   
                    'y_shift_for_surface_adj': obstruction2D_dict['y_shift_for_surface_adj'],
                }
       
        # Only for Lithological Domain from Obstuction3D
        self_config['obstruction_overlap'] = self.obstruction_overlap
        return self_config

    @classmethod
    def from_config(cls, config_dict):
        if not isinstance(config_dict, dict):
            raise TypeError("Expected a dictionary.")
        
        try:
            obj = cls.__new__(cls)
            discretizedDomain2D = DiscretizedDomain2D.from_config(config_dict['domain'])
            name = config_dict['name']
            obj.domain = discretizedDomain2D
            obj.name = name
            
            # if config_dict['gwt_depth'] == 'None':
            #     config_dict['gwt_depth'] = None
            obj.gwt_depth = config_dict['gwt_depth']
            
            if config_dict['init_domain'] is None:
                obj.init_domain = None
            else:
                obj.init_domain = DiscretizedDomain2D.from_config(config_dict['init_domain'])

            obj.interface_config_revision_id = config_dict['interface_config_revision_id']
            obj.lithological_matrix = config_dict['lithological_matrix']
            obj.lm_type = config_dict['lm_type']
            obj.merged_lit = config_dict['merged_lit']
            obj.obstruction_overlap = config_dict['obstruction_overlap']
            obj.lit_order = config_dict['lit_order']
            
            obj.obstruction2d_dict_list = []
            obstruction2D_instance_list = config_dict['obstruction2d_dict_list']
            if len(obstruction2D_instance_list)!=0:
                for _,obstruction2D_dict_raw in obstruction2D_instance_list.items():
                    obstruction2D_dict = {
                        'obstruction_inst': Obstruction2D.from_config(obstruction2D_dict_raw['obstruction_inst']),
                        'shift_ref2d_to_xy': obstruction2D_dict_raw['shift_ref2d_to_xy'],        
                        'added_prefix': obstruction2D_dict_raw['added_prefix'],   
                        'y_shift_for_surface_adj': obstruction2D_dict_raw['y_shift_for_surface_adj'],
                    }
                    obj.obstruction2d_dict_list.append(obstruction2D_dict)
            return obj

        except (KeyError, TypeError) as e:
            raise ValueError(f"Invalid config dictionary: {e}")
        
    # def __eq__(self, other):
    #     if not isinstance(other, LithologicalDomain2DReadOnly):
    #         return NotImplemented
        
        # units_check = self.units_config == other.units_config
        # spans_check = np.allclose(self._spans_in_domain_len_units, other._spans_in_domain_len_units)
        # dhs_check = np.allclose(self._dhs_in_domain_len_units, other._dhs_in_domain_len_units)
        # return (
        #     units_check
        #     and spans_check
        #     and dhs_check
        # )

