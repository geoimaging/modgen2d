import numpy as np

from geomodgen2d.interfaces_creator2d import AbstractInterfacesCreator2D
from geomodgen2d.discretized_domain2d import DiscretizedDomain2D
import geomodgen2d.general_functions as f

class SurfaceInterface2D(AbstractInterfacesCreator2D):
    def __init__(self, domain: DiscretizedDomain2D, remesh_interp_method = 'linear', rng=np.random.default_rng()):
        """
        Initializes the DiscretizedInterfaces2D instance (with only one interface) with given discretized domain2d instance.
        
        Parameters:
        domain2D : DiscretizedDomain2D
            The DiscretizedDomain2D instance describing the spans and dhs of the domain.
        remesh_interp_method: str
            Interpolation method when remeshing (default: 'linear')
        rnd_no: 
            Optional random number generator instance
        """
        n_interfaces = 1
        super().__init__(domain, n_interfaces=n_interfaces, remesh_interp_method=remesh_interp_method, rng=rng)

    def is_surface_interface(self):
        return True
    
    def seperate_surface_interface(self):
        """
        Seperate surface interface and return main and surface interface after seperating.
        Surface interface is the top interface.
        """
        raise ValueError("Already a SurfaceInterface2D. Hence, Surface interface cannot be seperated")
        
        
class DiscretizedInterfaces2D(AbstractInterfacesCreator2D):
    def __init__(self, domain: DiscretizedDomain2D, n_interfaces: int, remesh_interp_method = 'linear', rng=np.random.default_rng()):
        """
        Initializes the DiscretizedInterfaces2D instance with given discretized domain2d instance, and number of interfaces. 
        
        Parameters:
        domain2D : DiscretizedDomain2D
            The DiscretizedDomain2D instance describing the spans and dhs of the domain.
        n_interfaces: float, 
            Number of soil interfaces in the model
        remesh_interp_method: str
            Interpolation method when remeshing (default: 'linear')
        rnd_no: 
            Optional random number generator instance
        """
        super().__init__(domain, n_interfaces=n_interfaces, remesh_interp_method=remesh_interp_method, rng=rng)

    def is_surface_interface(self):
        return False
    
    def seperate_surface_interface(self):
        """
        Seperate surface interface and return main and surface interface after seperating.
        Surface interface is the top interface.
        """
        if self.n_interfaces == 0:
            raise ValueError("Surface interface cannot be seperated, provided the interface class has no interfaces. Need at least one to seperate.")
        
        surface_interface_instance = SurfaceInterface2D(self.domain, self.remesh_interp_method, self.rng)
        surface_interface_instance.set_interfaces_matrix(self.interfaces_matrix[:,0:1])
        
        soil_interface_instance = DiscretizedInterfaces2D(self.domain, self.n_interfaces-1, self.remesh_interp_method, self.rng)
        if soil_interface_instance.n_interfaces!=0:
            soil_interface_instance.set_interfaces_matrix(self.interfaces_matrix[:,1:])
            
        if self._locked:
            soil_interface_instance.lock_interfaces()
            surface_interface_instance.lock_interfaces()
            
        return soil_interface_instance, surface_interface_instance
    
    def get_interfaces_matrix_with_surface(self, surface_interface2d_instance: AbstractInterfacesCreator2D = None, surface_interface_method="pile"):
        """
        Get the interface matrix with surface included.
        """
        if self.check_if_overlapping_interfaces():
            raise ValueError("Overlapping interfaces exist. Please use processing at main interface before merging with surface.") 
        
        if surface_interface_method not in ['pile', 'erode']:
            raise ValueError(f"Methods can only be either 'pile' or 'erode'. Provided: {surface_interface_method}")

        if surface_interface2d_instance is None:
            surface_interface2d_instance = SurfaceInterface2D(self.domain, self.remesh_interp_method, self.rng)
            surface_interface2d_instance.set_interfaces_matrix(np.zeros_like(self.interfaces_matrix[:,0:1]))
         
        if not isinstance(surface_interface2d_instance, AbstractInterfacesCreator2D):
                raise TypeError("surface_interface2d_instance must be a subclass of AbstractInterfacesCreator2D instance, or None (if no interface).")
              
        if surface_interface2d_instance.n_interfaces != 1:
            raise ValueError(f"SurfaceInterfaces2D must have 1 interface only. Provided {surface_interface2d_instance.n_interfaces}")
        
        if self.remesh_interp_method != surface_interface2d_instance.remesh_interp_method:
            raise ValueError(f"Interpolation methods of this ('{self.remesh_interp_method}') and surfaceInterface2D instance ('{self.remesh_interp_method}') does not match.")
        ## Make sure domains dhs are consistent
        if not (
            len(self.domain.spans) == len(surface_interface2d_instance.domain.spans)
            and all(f.is_close(a, b) for a, b in zip(self.domain.spans,
                                                surface_interface2d_instance.domain.spans))
        ):
            raise ValueError(
                "The domains' spans are not consistent. "
                f"Lithological domain has spans {self.domain.spans}, "
                f"while surface interface has {surface_interface2d_instance.domain.spans}"
            )
            
        if surface_interface2d_instance.domain != self.domain:
            surface_interface2d_instance = surface_interface2d_instance.remesh_interface(self.domain.dhs[0], self.domain.dhs[1])
        
        surf_interface_matrix = surface_interface2d_instance.interfaces_matrix
        self_interface_matrix = self.interfaces_matrix
        n_interfaces = self.n_interfaces + 1
        new_instance = DiscretizedInterfaces2D(self.domain, n_interfaces, self.remesh_interp_method, self.rng)
        
        if surface_interface_method == 'erode':
            # Preserve the reference_point_x value's 1D profile if not None
            if self._reference_point_x is not None:
                lock_status = surface_interface2d_instance._locked
                init_zero_val = surface_interface2d_instance.interfaces_matrix[0,0]
                surface_interface2d_instance._locked = False #Temporary unlock
                
                surface_interface2d_instance.update_interfaces_depth([0], self._reference_point_x)
                surf_interface_matrix = surface_interface2d_instance.interfaces_matrix
                
                #Retrace back
                surface_interface2d_instance.update_interfaces_depth(init_zero_val, [init_zero_val])
                surface_interface2d_instance._locked = lock_status
            
            # Perform eroding            
            new_interface_matrix = np.concatenate((surf_interface_matrix, self_interface_matrix), axis=1) 
            
        else: #'pile'
            # Preserve the reference_point_x value's 1D profile if not None
            # 1D profile at all points are preserved when piled. so need for processing for that.
            
            # Perform piling
            surf_interface_matrix = surface_interface2d_instance.interfaces_matrix
            new_interface_matrix = np.concatenate((np.zeros_like(surf_interface_matrix), self_interface_matrix), axis=1) 
            surf_interface_matrix = np.ones_like(new_interface_matrix)*surf_interface_matrix
            new_interface_matrix += surf_interface_matrix
            
        # Making the top of the surface_interface the top of the model too.
        new_interface_matrix = _adjust_for_top_surface_interface(new_interface_matrix)
        
        new_instance.set_interfaces_matrix(new_interface_matrix)
        new_instance.processing_interface(prioritize_lower_interface=True)
        new_instance.lock_interfaces()
        new_instance._surface_included = True
        return new_instance

def _adjust_for_top_surface_interface(interfaces_matrix):
    top_interface = interfaces_matrix[:,0]
    top_depth = np.min(top_interface)
    
    # computing the shift
    shift_matrix = np.full_like(interfaces_matrix, -top_depth) 
    interfaces_matrix+=shift_matrix
    if np.abs(np.min(interfaces_matrix)) <= -1e-2:
        raise SyntaxError("The minimum should have been greater than 0.")
    return interfaces_matrix

def generate_interfaces_from_interfaces_settings_dict(domain: DiscretizedDomain2D, n_interfaces:int, interfaces_settings_dict:dict, rng, remesh_interp_method='linear'):
    """
    Generate soil and surface interfaces from a settings dictionary.
    
    This function creates instances of `DiscretizedInterfaces2D` for soil interfaces and, 
    optionally, surface interfaces depending on the 'surface_factor'.

    Parameters
    ----------
    domain : DiscretizedDomain2D
        Discretized domain instance describing the model domain.
    n_interfaces : int
        Number of soil interfaces to generate.
    interfaces_settings_dict : dict
        Dictionary defining interface generation settings.
        Example format:
        {
            'generator_settings_dict':{
                'generator_option':'uniform',   # options: 'uniform', 'normal', 'fbm'
                'max_dz_per_unit_length':4.5,   # Required for 'uniform'
                'std':2,                        # Required for 'normal'
                'H':1, 'length':1, 'method':1,  # Required for 'fbm'
            },
            'surface_factor':0.5,                # factor applied to surface interface
            'interfaces_depths_generation':'random', 
            'interfaces_depth_reference_point_x':None, 
            'filter_settings': {
                'filter_window_length':3, 
                'filter_polyorder':2,
            },
            'processing_settings': {
                'prioritize_lower_interface': True,
            }
        }
    rng : np.random.Generator
        Random number generator instance.
    remesh_interp_method : str, optional
        Interpolation method for remeshing, by default 'linear'.

    Returns
    -------
    soil_interface : DiscretizedInterfaces2D
        Generated soil interface instance.
    surface_interface : DiscretizedInterfaces2D or None
        Generated surface interface instance if surface_factor != 0, else None.
    
    """
    required_keys = ['generator_settings_dict', 'interfaces_depths_generation', 'interfaces_depth_reference_point_x']
    optional_keys = ['filter_settings', 'processing_settings', 'surface_factor']
    allowed_keys = set(required_keys + optional_keys)

    # 1. Check for missing required keys
    missing = set(required_keys) - interfaces_settings_dict.keys()
    if missing:
        raise KeyError(f"Missing required keys in interfaces_settings_dict: {missing}")

    # 2. Check for unknown keys
    unknown = interfaces_settings_dict.keys() - allowed_keys
    if unknown:
        raise KeyError(f"Unknown keys in interfaces_settings_dict: {unknown}")

    surface_factor = interfaces_settings_dict.get('surface_factor', 0)
    
    # Generate soil interface
    soil_interface = DiscretizedInterfaces2D(domain, n_interfaces, remesh_interp_method, rng)
    soil_interface = _get_soil_interface3d_form_dict(soil_interface, interfaces_settings_dict, 1)
    
    # Generate surface interface if needed
    surface_interface = None
    if surface_factor != 0:
        interfaces_settings_dict['interfaces_depths_generation'] = 'random' # Random even if anyother settings provided
        surface_interface = DiscretizedInterfaces2D(domain, 1, remesh_interp_method, rng)
        surface_interface = _get_soil_interface3d_form_dict(surface_interface, interfaces_settings_dict, surface_factor)
    
    return soil_interface, surface_interface
        
def _get_soil_interface3d_form_dict(raw_interface_instance: AbstractInterfacesCreator2D, interfaces_settings_dict:dict, surface_factor=1):
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
    raw_interface_instance : AbstractInterfacesCreator2D
        Interface instance to configure.
    interfaces_settings_dict : dict
        Dictionary containing generator, filter, and processing settings.
    surface_factor : float, optional
        Factor to scale the generator settings for surface interfaces, by default 1.

    Returns
    -------
    AbstractInterfacesCreator2D
        Configured and locked interface instance.
    """
    filter_settings_dict = interfaces_settings_dict.get('filter_settings_dict', None)
    processing_settings = interfaces_settings_dict.get('processing_settings', None)
    
    #Step 1: Generate Rough Interfaces
    generator_settings_dict = interfaces_settings_dict['generator_settings_dict']
    
    if 'generator_option' not in generator_settings_dict:
        raise ValueError("'generator_option' must exist in generator_settings_dict.")
    
    random_generator_option = generator_settings_dict['generator_option']
    if random_generator_option=='uniform':
        missing = set(['max_dz_per_unit_length']) - generator_settings_dict.keys()
        if missing:
            raise KeyError(f"Missing required keys in generator_settings_dict for generator_option{random_generator_option}: {missing}")
        
    elif random_generator_option=='normal':
        missing = set(['stdev_in_unit_length']) - generator_settings_dict.keys()
        if missing:
            raise KeyError(f"Missing required keys in generator_settings_dict for generator_option{random_generator_option}: {missing}")
    
    elif random_generator_option=='fbm':
        missing = set(['H', 'length', 'method']) - generator_settings_dict.keys()
        if missing:
            raise KeyError(f"Missing required keys in generator_settings_dict for generator_option{random_generator_option}: {missing}")
    else:
        raise ValueError("random_generator_options can only be either 'uniform', or 'normal', or 'fbm'")
        
    raw_interface_instance.generate_rough_interfaces(generator_settings_dict, surface_scaling_factor=surface_factor)
    
    #Step 2: Filter
    if filter_settings_dict is not None:
        raw_interface_instance.filtering_interface(**filter_settings_dict)
        
    #Step 3: Update Interface Depth
    zs = raw_interface_instance.get_reference_points_zs(interface_z_references=interfaces_settings_dict['interfaces_depths_generation'])
    raw_interface_instance.update_interfaces_depth(zs,interfaces_settings_dict['interfaces_depth_reference_point_x'])  # None means get one from
    
    # Step 4: Processing
    if processing_settings is not None:
        raw_interface_instance.processing_interface(**processing_settings)
    
    # Step 5: Locking
    raw_interface_instance.lock_interfaces()
    
    return raw_interface_instance       
