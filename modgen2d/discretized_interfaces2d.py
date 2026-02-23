"""
Discretized 2D geological interfaces.

Defines utilities for generating, processing, remeshing,
and visualizing soil and surface interfaces over a
discretized 2D domain.
"""

import warnings

import numpy as np
import scipy

from modgen2d.discretized_domain2d import DiscretizedDomain2D
from modgen2d.rough_interface_creator2d import AbstractRoughInterfaceCreator
import modgen2d.general_functions as f
import matplotlib.pyplot as plt

class DiscretizedInterfaces2D:
    """
    Discretized 2D soil and surface interfaces.

    Interfaces are defined on a discretized 2D domain and may represent soil–soil or surface–soil boundaries.

    Once locked, the instance becomes immutable.
    """
    
    def __init__(self, domain: DiscretizedDomain2D, n_soil_layers: int, generate_surface:bool, rough_interface_generator_scale:list=None, remesh_interp_method = 'linear', rng=np.random.default_rng()):
        """
        Initializes the 'InterfaceCreator' class instance. 
        
        Parameters
        ----------
        domain : DiscretizedDomain2D
            The DiscretizedDomain2D instance describing the spans and dhs of the domain.
        n_soil_layers: int
            Number of soil layers.
        generate_surface:bool
            Whether a surface interface is present.
        rough_interface_generator_scale : list, optional
            Roughness scaling factors per interface.
        remesh_interp_method : str, optional
            Interpolation method used during remeshing. (default: 'linear')
        rng : numpy.random.Generator, optional
            Random number generator.
        """
        n_soil_layers = int(n_soil_layers)
        blank_interfaces = np.ones((domain.interface_shape[0], 
                                     n_soil_layers,
                                     )) * np.nan
        
        self.domain = domain
        self.interfaces_matrix = blank_interfaces
        # check shape of interfaces is compatible
        if domain.interface_shape[:-2] != self.interfaces_matrix.shape[:-2]:
            msg = f"Shape of domain {domain.interface_shape[:-2]} and "
            msg += f"interfaces {self.interfaces_matrix.shape[:-2]} are not the same."
            raise ValueError(msg)
        
        self.n_soil_layers = n_soil_layers
        self.generate_surface = generate_surface
        self.set_rough_interface_generator_scale(rough_interface_generator_scale)
        
        self.n_soil_soil_interfaces = n_soil_layers-1
        self.remesh_interp_method = remesh_interp_method
        self.rng = rng
        self._locked = False #Don't allow change in generated interfaces.
        self._reference_point_x = None #Reference point x used during updating interfaces,
        self._simulate_erosion_curr = True # Default: Changed as per arg. in .processing_interface(simulate_erosion=____)
        self._adjust_top_surface_zero = False # Sent to True if used .adjust_top_of_surface_interface_to_zero is used ever.

    @property
    def shape(self):
        return (len(self.domain.get_interface_x_centers), self.n_soil_layers)

    def get_rough_interface_generator_scale(self):
        return self.rough_interface_generator_scale
        
    def set_rough_interface_generator_scale(self, rough_interface_generator_scale):
        if rough_interface_generator_scale is None:
            rough_interface_generator_scale = [int(self.generate_surface), 1]
        
        rough_interface_generator_scale = np.asarray(rough_interface_generator_scale, dtype=float)
        if rough_interface_generator_scale.ndim != 1:
            raise ValueError("rough_interface_generator_scale must be a 1-D array or scalar")

        if len(rough_interface_generator_scale)==1 and self.n_soil_layers>1 and rough_interface_generator_scale[0]==0:
            warnings.warn(f"rough_interface_generator_scale is [0] and hence only horizontal interfaces will be created for all {self.n_soil_layers} if not corrected.")
    
        adj_rough_interface_generator_scale = np.full(self.n_soil_layers, rough_interface_generator_scale[-1], dtype=float)
        min_len = min(len(rough_interface_generator_scale), self.n_soil_layers)
        adj_rough_interface_generator_scale[:min_len] = rough_interface_generator_scale[:min_len]
        
        self.check_rough_interface_generator_scale(adj_rough_interface_generator_scale)
        self.rough_interface_generator_scale = adj_rough_interface_generator_scale    
        
    def is_surface_interface(self):
        return self.n_soil_layers==1
    
    def check_rough_interface_generator_scale(self, rough_interface_generator_scale):
        rough_interface_generator_scale = np.asarray(rough_interface_generator_scale, dtype=float)
        AbstractRoughInterfaceCreator.check_rough_interface_generator_scale(rough_interface_generator_scale)
        if not self.generate_surface and rough_interface_generator_scale[0]!=0:
            raise ValueError(f"Models with no/hz surface must have first element on rough_interface_generator_scale as 0. Provided {rough_interface_generator_scale[0]}")
        
        if len(rough_interface_generator_scale) != self.n_soil_layers:
            raise ValueError("The adjusted length of rough_interface_generator_scale must be equal to n_soil_layers. Try re setting the scale.")
    
    def lock_interfaces(self):
        """
        Lock the interfaces to prevent further modification.

        Raises
        ------
        ValueError
            If interfaces contain NaNs, overlap, or violate
            surface constraints.
        """
        if np.isnan(self.interfaces_matrix).any():
            raise ValueError("Interfaces_matrix contains NaN values.")
        if self.check_if_overlapping_interfaces()[0]:
            warnings.warn(f"Overlapping interfaces exist. Using .processing_interface(simulate_erosion = {self._simulate_erosion_curr})")
            self.processing_interface() 
        if not self.check_if_surface_is_okay():
            raise ValueError("Surface must have minimum value zero. Use ._adjust_for_top_surface_interface if needed.") 
        
        if self._adjust_top_surface_zero:
            self.adjust_top_of_surface_interface_to_zero()
        else:
            if self.generate_surface:
                warnings.warn("The top of surface_interface is not adjusted to zero. Use .adjust_top_of_surface_interface_to_zero() at least once for auto adjust everytime.")
        
        self._locked = True
    
    def check_if_surface_is_okay(self):
        return not np.abs(np.min(self.interfaces_matrix)) <= -1e-2
        
    def print(self):
        """Prints the dimensions of the boundary array and its content."""
        print(f"N_x_coord (includes 2 extra points at the edges) = {self.interfaces_matrix.shape[0]}, N_interfaces = {self.interfaces_matrix.shape[1]}")
        print("Interface Matrix (Transposed) \n", self.interfaces_matrix.T) #Transpose as formatted already in x,z. Numpy pretty prints for z,x

    def replace_top_surface(self, surface_interface_instance:"DiscretizedInterfaces2D", method='pile') -> "DiscretizedInterfaces2D":
        """
        Get the interface matrix with surface included.
        ## scaling factor also replaced.
        """
        # if self.check_if_overlapping_interfaces():
        #     raise ValueError("Overlapping interfaces exist. Please use processing at main interface before adding the top surface.") 
        
        if method not in ['pile', 'erode']:
            raise ValueError(f"Methods can only be either 'pile' or 'erode'. Provided: {method}")

        if not isinstance(surface_interface_instance, DiscretizedInterfaces2D):
            raise TypeError("surface_interface_instance must be a DiscretizedInterfaces2D class or its subclass.")
        
        if not isinstance(surface_interface_instance, DiscretizedInterfaces2D):
            raise TypeError("surface_interface_instance must be a DiscretizedInterfaces2D class or its subclass.")
         
        if surface_interface_instance.n_soil_layers != 1:
            raise ValueError(f"surface_interface_instance must have 1 interface (soil layer) only. Provided {surface_interface_instance.n_soil_layers}")
        
        if np.min(surface_interface_instance.interfaces_matrix)!=0:
            raise ValueError(f"The surface_interface_instance must have the min value of its inteface matrix as 0. Use ._adjust_for_top_surface_interface first if needed.")
        
        if self.remesh_interp_method != surface_interface_instance.remesh_interp_method:
            raise ValueError(f"Interpolation methods of this ('{self.remesh_interp_method}') and surface_interface_instance ('{surface_interface_instance.remesh_interp_method}') does not match.")
        ## Make sure domains dhs are consistent
        if not (
            len(self.domain.spans) == len(surface_interface_instance.domain.spans)
            and all(f.is_close(a, b) for a, b in zip(self.domain.spans,
                                                surface_interface_instance.domain.spans))
        ):
            raise ValueError(
                "The domains' spans are not consistent. "
                f"Lithological domain has spans {self.domain.spans}, "
                f"while surface interface has {surface_interface_instance.domain.spans}"
            )
            
        if surface_interface_instance.domain != self.domain:
            surface_interface_instance = surface_interface_instance.remesh_interface(self.domain.dhs[0], self.domain.dhs[1])
        
        surf_interface_matrix = surface_interface_instance.interfaces_matrix
        self_interface_matrix = self.interfaces_matrix
        new_instance = self.clone()
        
        
        ## Adjust the scaling factor and generate_surface
        new_instance.generate_surface = surface_interface_instance.generate_surface
        rough_interface_generator_scale = self.rough_interface_generator_scale
        
        new_scale = rough_interface_generator_scale.copy()
        new_scale[0] = surface_interface_instance.rough_interface_generator_scale[0]
        new_instance.set_rough_interface_generator_scale(new_scale)
        
        ## TODO: To discuss if worth it, or just do single creation and use erode. (Ie no surface distinction)
        if method == 'erode':
            # Preserve the reference_point_x value's 1D profile if not None
            if self._reference_point_x is not None:
                lock_status = surface_interface_instance._locked
                # init_zero_val = surface_interface2d_instance.interfaces_matrix[0,0]
                surface_interface_instance._locked = False #Temporary unlock
                
                # surface_interface2d_instance.update_interfaces_depth([0], self._reference_point_x)
                # surf_interface_matrix = surface_interface2d_instance.interfaces_matrix
                
                #Retrace back
                # surface_interface2d_instance.update_interfaces_depth([init_zero_val], init_zero_val)
                # surf_interface_matrix = surface_interface2d_instance.interfaces_matrix
                surface_interface_instance._locked = lock_status
            
            # Perform eroding            
            new_interface_matrix[:,0] = surf_interface_matrix[:,0]
            
        else: #'pile'
            # Preserve the reference_point_x value's 1D profile if not None
            # 1D profile at all points are preserved when piled. so need for processing for that.
            
            # Perform piling
            surf_interface_matrix = np.ones_like(self_interface_matrix)*surf_interface_matrix
            new_interface_matrix = self_interface_matrix + surf_interface_matrix
            
        # Making the top of the surface_interface the top of the model too.
        new_instance._locked =False
        new_instance.set_interfaces_matrix(new_interface_matrix)
        new_instance.processing_interface()
        new_instance.lock_interfaces()
        return new_instance

    def get_surface_and_subsurface_interfaces(self, relative_to_surface=False) -> "DiscretizedInterfaces2D":
        """
        Seperate surface interface and return main and surface interface after seperating.
        Surface interface is the top interface.
        """
        if self.n_soil_layers == 1:
            raise ValueError("Surface interface cannot be seperated, provided the interface class has no soil-soil interfaces. Need at least one to seperate.")
        
        surf_rough = self.rough_interface_generator_scale[0]
        surface_interface_instance = DiscretizedInterfaces2D(self.domain, 1, generate_surface=True, rough_interface_generator_scale=[surf_rough], remesh_interp_method=self.remesh_interp_method, rng=self.rng)
        surface_interface_instance.set_interfaces_matrix(self.interfaces_matrix[:,0:1])
        surface_interface_instance._adjust_top_surface_zero = self._adjust_top_surface_zero
        
        new_scale = self.rough_interface_generator_scale.copy()
        new_scale[0] = 0
        soil_interface_instance = DiscretizedInterfaces2D(self.domain, self.n_soil_layers, generate_surface=False, rough_interface_generator_scale=new_scale, remesh_interp_method=self.remesh_interp_method, rng=self.rng)
        new_interface = np.zeros_like(self.interfaces_matrix)
        
        if self.n_soil_layers > 1:
            if relative_to_surface:
                new_interface[:, 1:] = self.interfaces_matrix[:, 1:] - self.interfaces_matrix[:, :1]
            else:
                new_interface[:, 1:] = self.interfaces_matrix[:, 1:]

        soil_interface_instance.set_interfaces_matrix(new_interface)
        soil_interface_instance._simulate_erosion_curr = self._simulate_erosion_curr
        
        if self._locked:
            soil_interface_instance.lock_interfaces()
            surface_interface_instance.lock_interfaces()
            
        return soil_interface_instance, surface_interface_instance
    
    def set_interfaces_matrix(self, interfaces_matrix: np.ndarray):
        """
        Set the interface depth matrix.

        Parameters
        ----------
        interfaces_matrix : numpy.ndarray
            Interface depth matrix of shape
            ``(n_x_interface, n_soil_layers)``.

        Raises
        ------
        SystemError
            If the instance is locked.
        ValueError
            If shape, NaNs, or surface constraints are violated.
        """
        if self._locked:
            raise SystemError("This instance is fixed; no modifications allowed.")

        if isinstance(interfaces_matrix, list):
            interfaces_matrix = np.array(interfaces_matrix, dtype=float)

        if interfaces_matrix.shape != (len(self.domain.get_interface_x_centers), self.n_soil_layers):
            raise ValueError(f"Matrix shape mismatch. Note: x_centers_interfaces includes all x_centers in domain + 2 edges of x (i.e., 0 and x_span). Expected: {(len(self.domain.get_interface_x_centers), self.n_soil_layers)}. Got {interfaces_matrix.shape}")
        
        if np.isnan(interfaces_matrix).any():
            raise ValueError("Interfaces_matrix contains NaN values.")
        
        surface_is_zero = np.allclose(interfaces_matrix[:, 0], 0.0)
        if not self.generate_surface and not surface_is_zero:
            raise ValueError("When generate_surface=False, the surface interface must be zero everywhere.")

        self.interfaces_matrix = interfaces_matrix

    def generate_rough_interfaces(self, rough_interface_creator_instance: AbstractRoughInterfaceCreator):
        """
        Generate rough interfaces using a generator instance.

        Parameters
        ----------
        rough_interface_creator_instance : AbstractRoughInterfaceCreator
            Generator defining interface roughness.
        """
        nx, _ = self.interfaces_matrix.shape
        interfaces_matrix = np.zeros_like(self.interfaces_matrix)
        dx = self.domain.dhs[0]
        
        self.check_rough_interface_generator_scale(self.rough_interface_generator_scale)
        interfaces_matrix = rough_interface_creator_instance.generate_rough_interfaces(self.rough_interface_generator_scale, nx, dx=dx)
        self.set_interfaces_matrix(interfaces_matrix)
    
    def filtering_interface(self, filter_window_length=21, filter_polyorder=3):
        """
        Applies a Savitzky-Golay filter to smooth the interface.

        Parameters
        ----------
        filter_window_length: int
            Window size for the filter. If the value is zero, then it means no filtering.
        filter_polyorder: int
            Polynomial order for the filter
        """
        # interfaces_matrix = np.empty_like(self.interfaces_matrix)
        # if filter_window_length!=0:
        #     for i in np.arange(self.n_soil_layers):
        #         interfaces_matrix[:, i] = scipy.signal.savgol_filter(self.interfaces_matrix[:, i], window_length=filter_window_length, polyorder=filter_polyorder)#, window_length, polyorder
        # self.set_interfaces_matrix(interfaces_matrix)
        
        if filter_window_length!=0:
            interfaces_matrix = scipy.signal.savgol_filter(self.interfaces_matrix, window_length=filter_window_length, polyorder=filter_polyorder, axis=0)#, window_length, polyorder
        self.set_interfaces_matrix(interfaces_matrix)

    def get_reference_points_zs(self, method='random'):
        """
        Initializes interface points at reference coordinates. 
        
        Parameters
        ----------
        method: str 
            Mode of initialization ('equidistant', 'random')
        """
        span_z = self.domain.spans[1]

        # Getting the z-values at the reference point.
        if method == 'equidistant':
            # Equal thickness at y = 0  eg. [1/4,2/4,3/4] for n_soil_layers = 4
            reference_points_zs = np.arange(1, self.n_soil_layers) * span_z / self.n_soil_layers

        elif method == 'random':
            rndm_numbers = self.rng.random(self.n_soil_soil_interfaces)
            rndm_numbers.sort() ## TODO discuss sort vs random x vs other?
            reference_points_zs = rndm_numbers * span_z

        else:
            raise ValueError("method must either 'equidistant' or 'random'")

        return np.concatenate(([0.0], reference_points_zs))

    def update_interfaces_depth(self, reference_points_zs, reference_point_x=None):
        """
        Initializes interface points at reference coordinates. Shifting the boundaries' reference points so that they match the values in interface_init_points.

        Parameters
        ----------
        reference_points_zs : array_like
            Reference depths for each interface.
        reference_point_x: float, optional
            Reference x-coordinate for initialization. Will save the reference point, in case merged with surface (later).
            If None, first point in the x_centers.

        """
        interfaces_matrix = self.interfaces_matrix
        x_centers = self.domain.get_interface_x_centers
        reference_points_zs = np.asarray(reference_points_zs, dtype=float)
        
        if not np.issubdtype(reference_points_zs.dtype, np.number):
            raise ValueError("reference_points_zs must contain float values")

        if reference_points_zs.ndim != 1 or len(reference_points_zs)!=self.n_soil_layers:
            raise ValueError ( f"The provided no of reference points for interfaces ({len(reference_points_zs)}) != provided no of soil layers ({self.n_soil_layers}). Note: Surface interface cannot be given any reference value, so have first element 0.")

        if reference_points_zs[0] != 0:
            raise ValueError("reference_points_zs must have first element 0. Surface-soil interface is auto adjusted. So, all other references are relative to surface.")
            
        if not np.all(np.diff(reference_points_zs) >= 0):
            raise ValueError(f"reference_points_zs must be monotonically increasing. Provided {reference_points_zs}")
    
        # Locate reference point in grid
        if reference_point_x is None: 
            ref_idx = 0 ## TODO: Randomize here??
            reference_point_x = x_centers[ref_idx]
        elif not isinstance(reference_point_x, (int, float)):
            raise ValueError("reference_point_x must be a number")
        elif reference_point_x < x_centers[0] or reference_point_x > x_centers[-1]:   
            edge = x_centers[0] if reference_point_x < x_centers[0] else x_centers[-1]
            # warn if reference point is not on grid point
            msg = f"Requested position ({reference_point_x:.3f}) out of domain bound. "
            msg += f"Hence, setting to closest edge/bound ({edge:.3f})."
            warnings.warn(msg)
            reference_point_x = edge

        zs = range(self.n_soil_layers)
        
        if self.interfaces_matrix.shape[1] != 0:
            interp_ref_zs = f.remeshing_2D_matrix(x_old = x_centers, x_new = [reference_point_x],
                                                z_old = zs, z_new = zs, matrix_2d = self.interfaces_matrix, interp_method = self.remesh_interp_method)

            # computing the shift
            reference_points_zs+=interp_ref_zs[0,0]
            shift_z = reference_points_zs - interp_ref_zs[0,:]  
            shift_matrix = np.ones_like(interfaces_matrix) * shift_z[np.newaxis]
            interfaces_matrix += shift_matrix
        self.set_interfaces_matrix(interfaces_matrix)
        self._reference_point_x = reference_point_x

    def processing_interface(self, simulate_erosion=None):#, trim_interfaces=False):
        """
        Post-process interfaces to remove overlaps.

        Parameters
        ----------
        simulate_erosion : bool, optional
            If True, lower interfaces take priority.
        """
        #b_line_filtered_dict, zlim, top_priority=True):
        # Process 1: Limiting the boundaries to 0 and zlim, if trim_interfaces
        # Process 2: Interface crossing handling - Currently, priority given to lower interface (v3: option to reverse the priority)
        
        if simulate_erosion is None:
            simulate_erosion = self._simulate_erosion_curr
        
        b_array = self.interfaces_matrix[:, 1:]
        
        if not simulate_erosion:
            for i in np.arange(b_array.shape[-1]-2, -1, -1):
                b_array[:, i] = np.minimum(b_array[:, i], b_array[:, i+1])
        
        b_array = self.interfaces_matrix
        for i in np.arange(1, b_array.shape[-1]):
            b_array[:, i] = np.maximum(b_array[:, i], b_array[:, i-1])
        # if trim_interfaces:
        #     trim_z = self.domain.spans[1]
        # else:
        #     trim_z = max(self.interfaces_matrix)
            
        # b_array=np.clip(b_array, 0, trim_z) #clip between 0 and trim_z           
        self.set_interfaces_matrix(b_array)
        self._simulate_erosion_curr = simulate_erosion
    
    def check_if_overlapping_interfaces(self):
        diffs = np.diff(self.interfaces_matrix)

        # Not enough interfaces → no overlap possible
        if diffs.size == 0:
            return False, 0

        has_overlap = np.any(diffs < 0)      # zero is NOT overlap
        min_abs_diff = np.min(np.abs(diffs))

        return has_overlap, min_abs_diff

    def adjust_top_of_surface_interface_to_zero(self):
        interfaces_matrix = self.interfaces_matrix
        top_interface = interfaces_matrix[:,0]
        top_depth = np.min(top_interface)
        
        # computing the shift
        shift_matrix = np.full_like(interfaces_matrix, -top_depth) 
        interfaces_matrix+=shift_matrix
        if np.abs(np.min(interfaces_matrix)) <= -1e-2:
            raise SyntaxError("The minimum should have been greater than 0.")
        self.set_interfaces_matrix(interfaces_matrix) 
        self._adjust_top_surface_zero = True
     
    def plot(self, ax=None, plot_extents=True, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(figsize=[8,8])

        n_soil_layers = self.n_soil_layers
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
        
        
        for i in np.arange(n_soil_layers-1, -1, -1):
            if i == 0:
                linestyle = '--'
            else:
                linestyle = '-'
            ax.plot(self.domain.get_interface_x_centers,
                    self.interfaces_matrix[:, i],label=i,
                    linestyle=linestyle,
                    drawstyle=drawstyle,
                    **kwargs,
                   # color=color_code[i],
                   )

        if plot_extents:
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
        Return a remeshed copy of the interfaces.

        Parameters
        ----------
        new_dx : float
            New discretization in x-direction.
        new_dz : float, optional
            New discretization in z-direction.

        Returns
        -------
        DiscretizedInterfaces2D
            Remeshed interface instance.
        """
        if new_dz is None:
            new_dz = self.domain.dhs[1]
        
        new = self.clone()
        new_domain = self.domain.remesh(new_dx, new_dz)
        new.domain = new_domain
        
        if self.n_soil_layers != 0:
            zs = range(self.n_soil_layers)
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
            new._simulate_erosion_curr = self._simulate_erosion_curr
            new._adjust_top_surface_zero = self._adjust_top_surface_zero
            # Possible issue of crisscross after remeshing on some rare conditions.
            if new._locked:
                flag, min_val = new.check_if_overlapping_interfaces()
                if flag:
                    new._locked = False
                    new.processing_interface()
                    new.lock_interfaces()
                    if min_val > 1e-5:
                        warnings.warn("Overlapping interfaces found after remesh; Applied default erosion processing at the edges to correct them. This should not affect most models.")
        else:
            new.interfaces_matrix = self.interfaces_matrix.copy()
            
        # if self._locked:
        #     new.lock_interfaces()
        return new
    
    @property
    def get_config(self):
        """Return class configuration."""
        domain_config = self.domain.get_config
        return {
            'domain': domain_config,
            'generate_surface': self.generate_surface,
            'n_soil_layers':self.n_soil_layers,
            'interfaces_matrix': self.interfaces_matrix,
            'remesh_interp_method':self.remesh_interp_method,
            'rough_interface_generator_scale':self.rough_interface_generator_scale,
            'rng_state':self.rng.bit_generator.state,
            'locked':self._locked,
            'reference_point_x': self._reference_point_x,
            'simulate_erosion_curr':self._simulate_erosion_curr,
            'adjust_top_surface_zero': self._adjust_top_surface_zero
        }
        
    @classmethod
    def from_config(cls, config_dict):
        """Create 'DiscretizedInterfaces2D' class instance from a configuration dictionary."""
        if not isinstance(config_dict, dict):
            raise TypeError("Expected a dictionary.")
        try:
            discretizedDomain2D = DiscretizedDomain2D.from_config(config_dict['domain'])
            interfaces_matrix = np.array(config_dict['interfaces_matrix'])
            n_soil_layers, generate_surface = config_dict['n_soil_layers'], config_dict['generate_surface']
            rough_interface_generator_scale, remesh_interp_method = config_dict['rough_interface_generator_scale'], config_dict['remesh_interp_method']

            rng = np.random.default_rng()
            rng.bit_generator.state = config_dict['rng_state']

            obj = cls(discretizedDomain2D, n_soil_layers, generate_surface, rough_interface_generator_scale,
                       remesh_interp_method, rng)
            obj.set_interfaces_matrix(interfaces_matrix)
            obj._reference_point_x = config_dict['reference_point_x']
            obj._simulate_erosion_curr = config_dict['simulate_erosion_curr']
            obj._adjust_top_surface_zero = config_dict['adjust_top_surface_zero']
            
            if config_dict['locked']:
                obj.lock_interfaces()
            return obj
        
        except (KeyError, TypeError) as e:
            raise ValueError(f"Invalid config dictionary: {e}")
        
    # def __eq__(self, other, gen_params_check=False):
    #     if not isinstance(other, DiscretizedInterfaces2D):
    #         return NotImplemented
        
    #     domain_check  = self.domain == other.domain
    #     interfaces_check = f.safe_equal(self.interfaces_matrix, other.interfaces_matrix)
    #     n_soils_check    = f.safe_equal(self.n_soil_layers, other.n_soil_layers)
    #     n_soil_if_check  = f.safe_equal(self.n_soil_soil_interfaces, other.n_soil_soil_interfaces)
    #     remesh_check     = f.safe_equal(self.remesh_interp_method, other.remesh_interp_method)
    #     rough_check      = f.safe_equal(self.rough_interface_generator_scale, other.rough_interface_generator_scale)
    #     ref_check        = f.safe_equal(self._reference_point_x, other._reference_point_x)
    #     lock_check       = f.safe_equal(self._locked, other._locked)
    #     rng_check = self.rng.bit_generator.state == other.rng.bit_generator.state

    #     params_check = (rough_check and ref_check and lock_check and rng_check)
    #     if not gen_params_check:
    #         params_check = True
        
    #     return (
    #         domain_check
    #         and interfaces_check
    #         and n_soils_check
    #         and n_soil_if_check
    #         and remesh_check
    #         and params_check
    #     )



    