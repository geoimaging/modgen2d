from abc import ABC, abstractmethod

import numpy as np
from fbm import FBM

class AbstractRoughInterfaceCreator(ABC):
    def __init__(self, generator_params:dict, rng=np.random.default_rng()):
        """
        Initializes the InterfaceCreator instance with given discretized domain2d instance, and number of interfaces.
        
        Parameters:
        domain2D : DiscretizedDomain2D
            The DiscretizedDomain2D instance describing the spans and dhs of the domain.
        n_soil_layers: float, 
            Number of soil interfaces in the model
        surface_scaling_factor : float
            Scaling factor for interface amplitude.
        remesh_interp_method: str
            Interpolation method when remeshing (default: 'linear')
        rnd_no: 
            Optional random number generator instance
        """
        self.generator_params = generator_params
        self.rng = rng
    
    @staticmethod
    def check_rough_interface_generator_scale(rough_interface_generator_scale):
        if not np.issubdtype(rough_interface_generator_scale.dtype, np.number):
            raise ValueError("rough_interface_generator_scale must contain float values")
        
        if rough_interface_generator_scale.ndim != 1:
            raise ValueError("rough_interface_generator_scale must be one-dimensional")
        
        if len(rough_interface_generator_scale)<1:
            raise ValueError("There should be at least one number in rough_interface_generator_scale. Found none. Set the rough_interface_generator_scale again.")
        
    @abstractmethod #Main Purpose to not allow user to use this method directly
    def generate_rough_interfaces(self, rough_interface_generator_scale, nx, dx, **kwargs):
        """
        Generate rough interfaces.

        Parameters
        ----------
        abs_interface_creator_instance : AbstractInterfacesCreator2D
            Interface creator instance.

        dx : float or None
            Spatial discretization. Optional for methods that do not require it.

        **kwargs
            Method-specific parameters (e.g., random generator settings).
        """
        pass
    
class UniformInterfaceGen(AbstractRoughInterfaceCreator):
    def __init__(self, max_dz_per_unit_length, rng=np.random.default_rng()):
        generator_params = {'max_dz_per_unit_length': max_dz_per_unit_length}
        super().__init__(generator_params, rng)
       
    def generate_rough_interfaces(self, rough_interface_generator_scale, nx, dx):
        rough_interface_generator_scale = np.asarray(rough_interface_generator_scale, dtype=float)
        self.check_rough_interface_generator_scale(rough_interface_generator_scale)
        n_soil_layers = len(rough_interface_generator_scale)
        
        interfaces_matrix = np.zeros((nx, n_soil_layers))
    
        base_max_dz = self.generator_params['max_dz_per_unit_length']
        z_max_change_per_dx = base_max_dz * dx

        rnd_numbers = (self.rng.random((nx-1, n_soil_layers)) - 0.5) * 2 #Numbers ranging from 1 and -1
        dz = rnd_numbers * z_max_change_per_dx
        dz *= rough_interface_generator_scale[:n_soil_layers]


        interfaces_matrix[1:, :] = dz
        interfaces_matrix = np.cumsum(interfaces_matrix, axis=0)

        return interfaces_matrix
         
class NormalInterfaceGen(AbstractRoughInterfaceCreator):
    def __init__(self, stdev_in_unit_length, rng=np.random.default_rng()):
        generator_params = {'stdev_in_unit_length': stdev_in_unit_length}
        super().__init__(generator_params, rng)
       
    def generate_rough_interfaces(self, rough_interface_generator_scale, nx, dx):
        rough_interface_generator_scale = np.asarray(rough_interface_generator_scale, dtype=float)
        self.check_rough_interface_generator_scale(rough_interface_generator_scale)
        
        n_soil_layers = len(rough_interface_generator_scale)
        interfaces_matrix = np.zeros((nx, n_soil_layers))

        mean = 0
        sigma_1m = self.generator_params['stdev_in_unit_length']
        sigma_dx = sigma_1m * np.sqrt(dx)  #Standard deviation grows with sqrt distance
        
        dz = self.rng.normal(loc=mean, scale=sigma_dx, size=(nx-1, n_soil_layers)) #Numbers ranging with mean 0
        dz *= rough_interface_generator_scale[:n_soil_layers]
            
        interfaces_matrix[1:, :] = dz
        interfaces_matrix = np.cumsum(interfaces_matrix, axis=0)
        return interfaces_matrix
    
class FBMInterfaceGen(AbstractRoughInterfaceCreator):
    def __init__(self, H, length, method, rng=np.random.default_rng()):
        generator_params = {'H': H,
                            'length': length,
                            'method': method}
        super().__init__(generator_params, rng)
       
    def generate_rough_interfaces(self, rough_interface_generator_scale, nx, dx='ignored'):
        rough_interface_generator_scale = np.asarray(rough_interface_generator_scale, dtype=float)
        self.check_rough_interface_generator_scale(rough_interface_generator_scale)
        
        n_soil_layers = len(rough_interface_generator_scale)
        interfaces_matrix = np.zeros((nx, n_soil_layers))

        H = self.generator_params['H']
        L = self.generator_params['length'] # *surface_scaling_factor While this gives approx scaling
        method = self.generator_params['method']
    
        n = nx - 1
        for j in range(n_soil_layers):
            scale = rough_interface_generator_scale[j]
            #generates n+1 data ie n increments
            rnd_layer = FBM(n=n, hurst=H, length=L, method=method).fbm() * scale  
            interfaces_matrix[:, j]= rnd_layer
        return interfaces_matrix
    
    