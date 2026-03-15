import numpy as np

from modgen2d.discretized_interfaces2d import DiscretizedInterfaces2D
from modgen2d.discretized_domain2d import DiscretizedDomain2D

class DiscretizedInterfaces2DFromDict(DiscretizedInterfaces2D):
    """Generate discretized 2D interfaces from a configuration dictionary."""
    def __init__(self, domain: DiscretizedDomain2D, n_soil_layers: int, interfaces_settings_dict:dict, remesh_interp_method = 'linear', rng=np.random.default_rng()):
        """
        Generate soil and surface interfaces from a settings dictionary.
        
        Parameters
        ----------
        domain : DiscretizedDomain2D
            Discretized domain instance describing the model domain.
        n_soil_layers : int
            Number of soil layers in the model.
        interfaces_settings_dict : dict
            Dictionary defining interface generation settings.

            Example format:
            {
                'generate_surface':True,
                'rough_interface_creator_instance':rough_interface_creator_instance: AbstractRoughInterfaceCreator
                'rough_interface_generator_scale':[0.2,1],          # factor applied to each interfaces. Note inteface0 is surface-soil interface. Will replicate last value, if n_layers>len(this list)
                'interfaces_depths_generation':'random',      # Can be 'random', 'normal', or np.ndarray (skips zs generation.)
                'interfaces_depth_reference_point_x':None, 
                'filter_settings': {
                    'filter_window_length':3, 
                    'filter_polyorder':2,
                },
                'processing_settings': {
                    'simulate_erosion': True,
                }
            }

        remesh_interp_method : str, optional
            Interpolation method for remeshing, by default 'linear'.
        rng : numpy.random.Generator, optional
            Random number generator.
        """

        required_keys = ['generate_surface', 'rough_interface_creator_instance', 'interfaces_depths_generation', 'interfaces_depth_reference_point_x']
        optional_keys = ['filter_settings', 'processing_settings', 'rough_interface_generator_scale', 'max_n_soil_layers_expected']
        allowed_keys = set(required_keys + optional_keys)

        # 1. Check for missing required keys
        missing = set(required_keys) - interfaces_settings_dict.keys()
        if missing:
            raise KeyError(f"Missing required keys in interfaces_settings_dict: {missing}")

        # 2. Check for unknown keys
        unknown = interfaces_settings_dict.keys() - allowed_keys
        if unknown:
            raise KeyError(f"Unknown keys in interfaces_settings_dict: {unknown}")

        generate_surface = interfaces_settings_dict['generate_surface']
        rough_interface_generator_scale = interfaces_settings_dict.get('rough_interface_generator_scale', None)
        
        super().__init__(domain, n_soil_layers, generate_surface, rough_interface_generator_scale, remesh_interp_method, rng)
        self._get_soil_interface2d_from_dict(interfaces_settings_dict)
        
    def _get_soil_interface2d_from_dict(self, interfaces_settings_dict:dict):
        """
        Configure a DiscretizedInterfaces2D instance from a settings dictionary.
        
        Performs the following steps:
        1. Generate rough interfaces according to generator settings and surface factor.
        2. Filter interfaces if filter settings are provided.
        3. Update interface depths using reference points.
        4. Apply post-processing if specified.
        5. Lock the interface to prevent further modification.

        Parameters
        ----------
        interfaces_settings_dict : dict
            Dictionary containing interface generator, filter, and processing settings.
        """
        filter_settings_dict = interfaces_settings_dict.get('filter_settings', None)
        processing_settings = interfaces_settings_dict.get('processing_settings', None)
        
        #Step 1: Generate Rough Interfaces
        rough_interface_creator_instance = interfaces_settings_dict['rough_interface_creator_instance']
        self.generate_rough_interfaces(rough_interface_creator_instance)
        
        #Step 2: Filter
        if filter_settings_dict is not None:
            self.filtering_interface(**filter_settings_dict)
            
        #Step 3: Update Interface Depth
        interfaces_depths_generation=interfaces_settings_dict['interfaces_depths_generation']
        if isinstance(interfaces_depths_generation, (np.ndarray, list)):
            zs = np.asarray(interfaces_depths_generation, dtype=float)
        else:
            zs = self.get_reference_points_zs(method=interfaces_depths_generation)
        self.update_interfaces_depth(zs,interfaces_settings_dict['interfaces_depth_reference_point_x'])  # None means get one from
        
        # Step 4: Processing and adjust for zero
        if processing_settings is not None:
            self.processing_interface(**processing_settings)
        
        self.adjust_top_of_surface_interface_to_zero()
        
        # Step 5: Locking
        self.lock_interfaces()
        
