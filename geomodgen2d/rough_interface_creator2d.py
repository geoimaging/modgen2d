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
        n_interfaces: float, 
            Number of soil interfaces in the model
        remesh_interp_method: str
            Interpolation method when remeshing (default: 'linear')
        rnd_no: 
            Optional random number generator instance
        """
        self.generator_params = generator_params
        self.rng = rng
        
    @abstractmethod #Main Purpose to not allow user to use this method directly
    def generate_rough_interfaces(self, nx, n_interfaces, dx=None, surface_scaling_factor=1, **kwargs):
        """
        Generate rough interfaces.

        Parameters
        ----------
        abs_interface_creator_instance : AbstractInterfacesCreator2D
            Interface creator instance.

        dx : float or None
            Spatial discretization. Optional for methods that do not require it.

        surface_scaling_factor : float
            Scaling factor for interface amplitude.

        **kwargs
            Method-specific parameters (e.g., random generator settings).
        """
        pass
    
class UniformInterfaceGen(AbstractRoughInterfaceCreator):
    def __init__(self, max_dz_per_unit_length, rng=np.random.default_rng()):
        generator_params = {'max_dz_per_unit_length': max_dz_per_unit_length}
        super().__init__(generator_params, rng)
       
    def generate_rough_interfaces(self, nx, n_interfaces, dx, surface_scaling_factor=1):
        interfaces_matrix = np.zeros((nx, n_interfaces))
        
        max_dz_per_unit_length = self.generator_params['max_dz_per_unit_length']*surface_scaling_factor
        z_max_change_per_dx = (max_dz_per_unit_length*dx)
        
        rnd_numbers = (self.rng.random(((nx-1), n_interfaces))-0.5)*2 #Numbers ranging from 1 and -1
        interfaces_matrix[1:, :] = rnd_numbers*z_max_change_per_dx
        interfaces_matrix = np.cumsum(interfaces_matrix, axis=0)
        return interfaces_matrix
         
class NormalInterfaceGen(AbstractRoughInterfaceCreator):
    def __init__(self, stdev_in_unit_length, rng=np.random.default_rng()):
        generator_params = {'stdev_in_unit_length': stdev_in_unit_length}
        super().__init__(generator_params, rng)
       
    def generate_rough_interfaces(self, nx, n_interfaces, dx, surface_scaling_factor=1):
        interfaces_matrix = np.zeros((nx, n_interfaces))

        mean = 0
        sigma_1m = self.generator_params['stdev_in_unit_length']
        sigma_dx = sigma_1m * np.sqrt(dx) * surface_scaling_factor  #Standard deviation grows with sqrt distance
        
        rnd_numbers = self.rng.normal(loc=mean, scale=sigma_dx, size=(nx-1, n_interfaces)) #Numbers ranging with mean 0
        interfaces_matrix[1:, :] = rnd_numbers
        interfaces_matrix = np.cumsum(interfaces_matrix, axis=0)
        return interfaces_matrix
    
class FBMInterfaceGen(AbstractRoughInterfaceCreator):
    def __init__(self, H, length, method, rng=np.random.default_rng()):
        generator_params = {'H': H,
                            'length': length,
                            'method': method}
        super().__init__(generator_params, rng)
       
    def generate_rough_interfaces(self, nx, n_interfaces, dx='ignored', surface_scaling_factor=1):
        interfaces_matrix = np.zeros((nx, n_interfaces))

        H = self.generator_params['H']
        L = self.generator_params['length'] # *surface_scaling_factor While this gives approx scaling
        method = self.generator_params['method']
    
        n = nx - 1
        for j in range(n_interfaces):
            #generates n+1 data ie n increments
            rnd_layer = FBM(n=n, hurst=H, length=L, method=method).fbm() * surface_scaling_factor  #Better approach
            init_layer = interfaces_matrix[0, j]
            rnd_layer+=init_layer
            interfaces_matrix[:, j]= rnd_layer
        return interfaces_matrix
    
    