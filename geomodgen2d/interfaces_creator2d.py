from abc import ABC, abstractmethod
import warnings

import numpy as np
import scipy
from fbm import FBM

from geomodgen2d.discretized_domain2d import DiscretizedDomain2D
import geomodgen2d.general_functions as f
import matplotlib.pyplot as plt

class AbstractInterfacesCreator2D(ABC):
    def __init__(self, domain: DiscretizedDomain2D, n_interfaces: int, remesh_interp_method = 'linear', rng=np.random.default_rng()):
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
        if type(self) is AbstractInterfacesCreator2D:
            raise TypeError(
                "AbstractInterfacesCreator2D is an internal abstract base class "
                "and cannot be instantiated directly. Extend it instead."
            )

        n_interfaces = int(n_interfaces)
        blank_interfaces = np.ones((domain.interface_shape[0], 
                                     n_interfaces,
                                     )) * np.nan
        
        self.domain = domain
        self.interfaces_matrix = blank_interfaces
        # check shape of interfaces is compatible
        if domain.interface_shape[:-2] != self.interfaces_matrix.shape[:-2]:
            msg = f"Shape of domain {domain.interface_shape[:-2]} and "
            msg += f"interfaces {self.interfaces_matrix.shape[:-2]} are not the same."
            raise ValueError(msg)
        
        self.n_interfaces = n_interfaces
        self.n_layers = self.n_interfaces + 1
        self.remesh_interp_method = remesh_interp_method
        self.rng = rng
        self._locked = False #Don't allow change in generated interfaces.
        self._reference_point_x = None #Reference point x used during updating interfaces,
        self._surface_included = False

    @property
    def shape(self):
        return (len(self.domain.get_interface_x_centers), self.n_interfaces)
    
    def lock_interfaces(self):
        if np.isnan(self.interfaces_matrix).any():
            raise ValueError("Interfaces_matrix contains NaN values.")
        if self.check_if_overlapping_interfaces():
            raise ValueError("Overlapping interfaces exist. Please use processing at the end if needed.") 
        self._locked = True
    
    def print(self):
        """Prints the dimensions of the boundary array and its content."""
        print(f"N_x_coord (includes 2 extra points at the edges) = {self.interfaces_matrix.shape[0]}, N_interfaces = {self.interfaces_matrix.shape[1]}")
        print("Interface Matrix (Transposed) \n", self.interfaces_matrix.T) #Transpose as formatted already in x,z. Numpy pretty prints for z,x

    @abstractmethod #Main Purpose to not allow user to use this method directly
    def is_surface_interface(self):
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def seperate_surface_interface(self):
        raise NotImplementedError("Subclasses must implement this method")
    
    def set_interfaces_matrix(self, interfaces_matrix: np.ndarray):
        """
        Set the interfaces_matrix with a new matrix.
        
        Parameters:
        interfaces_matrix : numpy array: 
            A NumPy array of the same shape as the existing interfaces_matrix
        """
        if self._locked:
            raise SystemError("This instance is fixed; no modifications allowed.")

        if isinstance(interfaces_matrix, list):
            interfaces_matrix = np.array(interfaces_matrix, dtype=float)

        if self.is_surface_interface():
            if self.n_interfaces != 1:
                raise ValueError(f"is_surface_interface is marked True, but provided {self.n_interfaces} (Surface has only 1)")

        if interfaces_matrix.shape != (len(self.domain.get_interface_x_centers), self.n_interfaces):
            raise ValueError(f"Matrix shape mismatch. Note: x_centers_interfaces includes all x_centers in domain + 2 edges of x (i.e., 0 and x_span). Expected: {(len(self.domain.get_interface_x_centers), self.n_interfaces)}. Got {interfaces_matrix.shape}")
        
        if np.isnan(interfaces_matrix).any():
            raise ValueError("Interfaces_matrix contains NaN values.")
        
        self.interfaces_matrix = interfaces_matrix

    def generate_rough_interfaces(self, random_generator_settings_dict, surface_scaling_factor=1):
        """
        Generates a interface matrix with randomized interface points.

        Parameters:
        random_generator_option: str
            Method for random generation of interface. Options: 'uniform', 'normal', 'fbm'.

        random_generator_settings_dict: dict
            Dictionary containing all relevant settings for respective option.

            Option: a) 'uniform' (original), next interface point at x+del_x = depth at x + uniformly generated random no.
                random_generator_settings_dict keys:
                    1) 'max_dz_per_unit_length' required: maximum change in z allowed per unit_length
                            
            Option: b) 'normal', next interface point at x+del_x = depth at x + gaussian generated random no.
                random_generator_settings_dict keys:
                    1)'std'
                        
            Option: c) 'fbm', interface generated based on fractional brownian motion (Cannot be replicated??)
                random_generator_settings_dict keys:
                    1) 'H' : Hurst index (1/2) for classical brownian motion
                    2) 'method': Solver for fbm

        surface_scaling_factor:
            Factor by which magnitude is to be reduced (for surface generation; If intended to have smaller undulations.)
        """
        nx, _ = self.interfaces_matrix.shape
        interfaces_matrix = np.zeros_like(self.interfaces_matrix)
        
        random_generator_option = random_generator_settings_dict['generator_option']
        if random_generator_option=='uniform':
            max_dz_per_unit_length = random_generator_settings_dict['max_dz_per_unit_length']*surface_scaling_factor
            
            dx = self.domain.dhs[0]
            
            z_max_change_per_dx = (max_dz_per_unit_length*dx)
            rnd_numbers = (self.rng.random(((nx-1), self.n_interfaces))-0.5)*2 #Numbers ranging from 1 and -1
            interfaces_matrix[1:, :] = rnd_numbers*z_max_change_per_dx
            interfaces_matrix = np.cumsum(interfaces_matrix, axis=0)
            
        elif random_generator_option=='normal':
            mean = 0
            stdev = random_generator_settings_dict['stdev_in_unit_length']*surface_scaling_factor
            rnd_numbers = self.rng.normal(loc=mean, scale=stdev, size=(nx-1, self.n_interfaces)) #Numbers ranging with mean 0
            interfaces_matrix[1:, :] = rnd_numbers
            interfaces_matrix = np.cumsum(interfaces_matrix, axis=0)
        
        elif random_generator_option=='fbm':
            H = random_generator_settings_dict['H']
            L = random_generator_settings_dict['length'] # *surface_scaling_factor While this gives approx scaling
            method = random_generator_settings_dict['method']
            n = nx - 1
            for j in range(self.n_interfaces):
                #generates n+1 data ie n increments
                rnd_layer = FBM(n=n, hurst=H, length=L, method=method).fbm() * surface_scaling_factor  #Better approach
                init_layer = interfaces_matrix[0, j]
                rnd_layer+=init_layer
                interfaces_matrix[:, j]= rnd_layer
        else:
            raise ValueError("random_generator_options can only be either 'uniform', or 'normal', or 'fbm'")
        
        self.set_interfaces_matrix(interfaces_matrix)
    
    def filtering_interface(self, filter_window_length=21, filter_polyorder=3):
        """
        Applies a Savitzky-Golay filter to smooth the interface.

        Parameters:
        filter_window_length: int
            Window size for the filter. If the value is zero, then it means no filtering.
        filter_polyorder: int
            Polynomial order for the filter
        """
        # interfaces_matrix = np.empty_like(self.interfaces_matrix)
        # if filter_window_length!=0:
        #     for i in np.arange(self.n_interfaces):
        #         interfaces_matrix[:, i] = scipy.signal.savgol_filter(self.interfaces_matrix[:, i], window_length=filter_window_length, polyorder=filter_polyorder)#, window_length, polyorder
        # self.set_interfaces_matrix(interfaces_matrix)
        
        if filter_window_length!=0:
            interfaces_matrix = scipy.signal.savgol_filter(self.interfaces_matrix, window_length=filter_window_length, polyorder=filter_polyorder, axis=0)#, window_length, polyorder
        self.set_interfaces_matrix(interfaces_matrix)

    def get_reference_points_zs(self, interface_z_references='random'):
        """
        Initializes interface points at reference coordinates. Shifting the boundaries' reference points so that they match the values in interface_init_points.

        Parameters:
        interface_z_references: str or list
            Mode of initialization ('equidistant', 'random', or numPy array of floats)

        """
        span_z = self.domain.spans[1]

        # Getting the z-values at the reference point.
        if isinstance(interface_z_references, np.ndarray):
            if not np.issubdtype(interface_z_references.dtype, np.number):
                raise ValueError("init_interface must contain float values")
            
            if interface_z_references.ndim != 1:
                raise ValueError("init_interface must be one-dimensional")
            
            if self.n_interfaces != interface_z_references.shape[-1]:
                raise ValueError(f"The provided reference boundaries's #boundaries ({interface_z_references.shape[-1]}) != provided no of interfaces ({self.n_interfaces})")
            reference_points_zs = interface_z_references

        elif interface_z_references == 'equidistant':
            # Equal thickness at y = 0  eg. [1/4,2/4,3/4] for n_layers = 4
            reference_points_zs= [i*span_z/self.n_layers for i in np.arange(1, self.n_layers)]
            
        elif interface_z_references == 'random':
            # Randomized and sorted, directly randomized the layer points
            rndm_numbers = self.rng.random(self.n_interfaces)
            rndm_numbers.sort()
            reference_points_zs = rndm_numbers*span_z
        else:
            raise ValueError("interface_z_reference must be a NumPy array of float values, if not 'equidistant', or 'random'")

        return np.array(reference_points_zs)
            
    def update_interfaces_depth(self, reference_points_zs, reference_point_x=None):
        """
        Initializes interface points at reference coordinates. Shifting the boundaries' reference points so that they match the values in interface_init_points.

        Parameters:
        interface_z_references: str or list
            Mode of initialization ('equidistant', 'random', or numPy array of floats)
        reference_point_x: float 
            Reference x-coordinate for initialization. Will save the reference point, in case merged with surface (later).
            If None, first point in the x_centers.

        """
        interfaces_matrix = self.interfaces_matrix
        x_centers = self.domain.get_interface_x_centers
        
        if len(reference_points_zs)!=self.n_layers-1:
            raise ValueError ( f"The provided no of reference points for interfaces ({len(reference_points_zs)}) != provided no of interfaces ({self.n_layers-1})")

        # Locate reference point in grid
        if reference_point_x is None: 
            reference_point_x = x_centers[0]
        elif not isinstance(reference_point_x, (int, float)):
            raise ValueError("reference_point_x must be a number")
        elif reference_point_x < x_centers[0] or reference_point_x > x_centers[-1]:   
            edge = x_centers[0] if reference_point_x < x_centers[0] else x_centers[-1]
            # warn if reference point is not on grid point
            msg = f"Requested position ({reference_point_x:.3f}) out of domain bound. "
            msg += f"Hence, setting to closest edge/bound ({edge:.3f})."
            warnings.warn(msg)
            reference_point_x = edge

        zs = range(self.n_interfaces)
        
        if self.interfaces_matrix.shape[1] != 0:
            interp_ref_zs = f.remeshing_2D_matrix(x_old = x_centers, x_new = [reference_point_x],
                                                z_old = zs, z_new = zs, matrix_2d = self.interfaces_matrix, interp_method = self.remesh_interp_method)


            # computing the shift
            shift_z = reference_points_zs - interp_ref_zs[0,:]  
            shift_matrix = np.ones_like(interfaces_matrix) * shift_z[np.newaxis]
            interfaces_matrix += shift_matrix
        self.set_interfaces_matrix(interfaces_matrix)
        self._reference_point_x = reference_point_x

    def processing_interface(self, prioritize_lower_interface=True):#, trim_interfaces=False):
        """
        Processes boundaries to prevent overlapping and limit values.
        
        prioritize_lower_interface: Boolean flag to determine priority in overlapping layers. If true, implies natural erosion.
        """
        #b_line_filtered_dict, zlim, top_priority=True):
        # Process 1: Limiting the boundaries to 0 and zlim, if trim_interfaces
        # Process 2: Interface crossing handling - Currently, priority given to lower interface (v3: option to reverse the priority)
        b_array = self.interfaces_matrix
        n_interface = self.n_interfaces
        
        if prioritize_lower_interface:
            for i in np.arange(1, n_interface):
                b_array[:, i] = np.maximum(b_array[:, i], b_array[:, i-1])
        else:
            for i in np.arange(n_interface-2, -1, -1):
                b_array[:, i] = np.minimum(b_array[:, i], b_array[:, i+1])
        
        # if trim_interfaces:
        #     trim_z = self.domain.spans[1]
        # else:
        #     trim_z = max(self.interfaces_matrix)
            
        # b_array=np.clip(b_array, 0, trim_z) #clip between 0 and trim_z           
        self.set_interfaces_matrix(b_array)
    
    def check_if_overlapping_interfaces(self):
        return np.sum(np.diff(self.interfaces_matrix)<0) > 0
       
    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=[8,8])

        n_interface = self.n_interfaces
        x_ranges = self.domain.get_interface_x_centers
        span_z = self.domain.spans[1]
        span_x = self.domain.spans[0]
        
        remesh_tech = self.remesh_interp_method
        if remesh_tech == 'nearest':
            drawstyle = 'steps-mid'
        elif remesh_tech == 'linear':
            drawstyle = 'default'
        else:
            warnings.warn(f"Interfaces might not reflect the exact interpolation in the plots except for 'linear' and 'nearest'. Provided {remesh_tech}.")
        
        
        for i in np.arange(n_interface-1, -1, -1):
            ax.plot(self.domain.get_interface_x_centers,
                    self.interfaces_matrix[:, i],label=i,
                    linestyle='-',
                    drawstyle=drawstyle,
                   # color=color_code[i],
                   )

        ax.plot([0,span_x], [0,0], 'k--',[0,span_x], [span_z, span_z],'k--')
        ax.set(ylim=[span_z, 0],
               xlim=[self.domain.get_interface_x_centers[0],self.domain.get_interface_x_centers[-1]],
               xlabel='Distance',
               ylabel='Depth',
               )
        ax.axis('scaled')
        return ax

    def clone(self):
        """
        Create a shallow clone of this object, copying all attributes
        except ones explicitly remeshed later.
        """
        new = object.__new__(self.__class__)
        new.__dict__ = self.__dict__.copy()
        return new

    def remesh_interface(self, new_dx, new_dz=None):
        """
        Returns a **new instance** of this interface class with remeshed coordinates,
        without modifying self.

        Parameters:
        new_dx, new_dz: float
            Refined spacing
        """
        
        if new_dz is None:
            new_dz = self.domain.dhs[1]
        
        new = self.clone()
        new_domain = self.domain.remesh(new_dx, new_dz)
        new.domain = new_domain
        
        if self.n_interfaces != 0:
            zs = range(self.n_interfaces)
            new_interfaces = f.remeshing_2D_matrix(
                x_old=self.domain.get_interface_x_centers,
                x_new=new_domain.get_interface_x_centers,
                z_old=zs,
                z_new=zs,
                matrix_2d=self.interfaces_matrix,
                interp_method=self.remesh_interp_method
            )
            # new_instance.set_interfaces_matrix(new_interfaces)
            new.interfaces_matrix = new_interfaces
            
            # Possible issue of crisscross after remeshing on some rare conditions.
            if new._locked:
                if new.check_if_overlapping_interfaces:
                    new._locked = False
                    new.processing_interface()
                    new.lock_interfaces()
                    warnings.warn("Overlapping interfaces found after remesh; Applied default erosion processing at the edges to correct them. This should not affect most models.")
        else:
            new.interfaces_matrix = self.interfaces_matrix.copy()
            
        # if self._locked:
        #     new.lock_interfaces()
        return new
    