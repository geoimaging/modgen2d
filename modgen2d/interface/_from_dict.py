import numpy as np

from ._main import DiscretizedInterfaces2D
from modgen2d.discretized_domain2d import DiscretizedDomain2D
from .depth_updaters import RandomDepthUpdater, OneBoreholeDepthUpdater, EquidistantDepthUpdater
from .interface_smoother import SavGol2DSmoother

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
                'rough_interface_generator_instance':rough_interface_generator_instance: AbstractRoughInterfaceCreator
                'savgol2d_smoother_settings': {
                    'filter_window_length':3, 
                    'filter_polyorder':2,
                },
                'interfaces_depths_updater':'random',      # Can be 'random', 'equidistant', or np.ndarray (skips zs generation.). Default: 'random',
                'interfaces_depth_reference_point_x':None,  # Default: None,
                'overlapping_resolver_technique': 'erosion', # Options: 'erosion', 'reverse_erosion'. Default: 'erosion'
                'adjust_surface_top_to_zero': True,  # Default: True
                }
            }

        remesh_interp_method : str, optional
            Interpolation method for remeshing, by default 'linear'.
        rng : numpy.random.Generator, optional
            Random number generator.
        """

        required_keys = ['generate_surface', 'rough_interface_generator_instance', 'interfaces_depths_updater', 'interfaces_depth_reference_point_x']
        optional_keys = ['savgol2d_smoother_settings', 'overlapping_resolver_technique', 'adjust_surface_top_to_zero', 'max_n_soil_layers_expected']
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
        
        super().__init__(domain, n_soil_layers, generate_surface, remesh_interp_method, rng)
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
        savgol2d_smoother_settings_dict = interfaces_settings_dict.get('savgol2d_smoother_settings', None)
        overlapping_resolver_technique = interfaces_settings_dict.get('overlapping_resolver_technique', 'erosion')
        adjust_surface_top_to_zero = interfaces_settings_dict.get('adjust_surface_top_to_zero', True)
        
        #Step 1: Generate Rough Interfaces
        # factor applied to each interfaces. Note inteface0 is surface-soil interface. Will replicate last value, if n_layers>len(this list)
        rough_interface_generator_instance =  interfaces_settings_dict.get('rough_interface_generator_instance')
        
        #Step 2: Filter
        if savgol2d_smoother_settings_dict is not None:
            smoother = SavGol2DSmoother(**savgol2d_smoother_settings_dict)
            
        #Step 3: Update Interface Depth
        interfaces_depths_updater=interfaces_settings_dict['interfaces_depths_updater']
        ref_x = interfaces_settings_dict['interfaces_depth_reference_point_x']
        if isinstance(interfaces_depths_updater, (np.ndarray, list)):
            zs = np.asarray(interfaces_depths_updater, dtype=float)
            depth_updater = OneBoreholeDepthUpdater(zs, ref_x)
        elif interfaces_depths_updater == 'random':
            depth_updater = RandomDepthUpdater(ref_x)
        elif interfaces_depths_updater == 'equidistant':
            depth_updater = EquidistantDepthUpdater(ref_x)
        
        # Step 4: Processing and adjust for zero
        self.apply_default_pipeline(rough_interface_generator_instance, smoother, depth_updater, overlapping_resolver_technique, adjust_surface_top_to_zero)
        
        # Step 5: Locking
        self.lock_interfaces()
        
