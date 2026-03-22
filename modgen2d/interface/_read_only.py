"""
Discretized 2D geological interfaces.

Defines utilities for generating, processing, remeshing,
and visualizing soil and surface interfaces over a
discretized 2D domain.
"""

import warnings

import numpy as np
from modgen2d.discretized_domain2d import DiscretizedDomain2D
import modgen2d.general_functions as f
import matplotlib.pyplot as plt

class DiscretizedInterfaces2DReadOnly:
    """
    ReadOnly class for Discretized 2D soil and surface interfaces. (all but apply.)

    Interfaces are defined on a discretized 2D domain and may represent soil–soil or surface–soil boundaries.

    Once locked, the instance becomes immutable.
    """
    
    def __init__(self, domain: DiscretizedDomain2D, n_soil_layers: int, generate_surface:bool, remesh_interp_method = 'linear', rng=np.random.default_rng()):
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
        roughness_multiplier : list, optional
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
        self.n_soil_soil_interfaces = n_soil_layers-1
        self.remesh_interp_method = remesh_interp_method
        self.rng = rng
        self._locked = False #Don't allow change in generated interfaces.
        self._ref_x = None #Reference point x used during updating interfaces,
        self._overlap_resolving_technique = True # Default: Changed as per arg. in .processing_interface(simulate_erosion=____)
        self._adjust_top_surface_zero = False # Sent to True if used .adjust_top_of_surface_interface_to_zero is used ever.
        self._adj_roughness_multipliers = None # Replaced with actual adj. multiplier once rough generator is used.

    @property
    def shape(self):
        return (len(self.domain.get_interface_x_centers), self.n_soil_layers)

    def is_surface_interface(self):
        return self.n_soil_layers==1
    
    def check_if_surface_is_okay(self):
        return not np.abs(np.min(self.interfaces_matrix)) <= -1e-2
    
    def lock_interfaces(self):
        """
        Lock the interfaces to prevent further modification.

        Raises
        ------
        ValueError
            If interfaces contain NaNs, overlap, or violate
            surface constraints.
        """
        self._locked = False
        if np.isnan(self.interfaces_matrix).any():
            raise ValueError("Interfaces_matrix contains NaN values.")
        if self.check_if_overlapping_interfaces()[0]:
            warnings.warn(f"Overlapping interfaces exist. Using .resolving_overlapped_interfaces(overlap_resolving_technique = {self._overlap_resolving_technique})")
            self.resolving_overlapped_interfaces() 
        # if not self.check_if_surface_is_okay():
        #     raise ValueError("Surface must have minimum value zero. Use ._adjust_for_top_surface_interface if needed.") 
        
        if self._adjust_top_surface_zero:
            self.adjust_top_of_surface_interface_to_zero()
        else:
            if self.generate_surface:
                warnings.warn("The top of surface_interface is not adjusted to zero. RECOMMENDED to use .adjust_top_of_surface_interface_to_zero().")
        
        self._locked = True
    
    def print(self):
        """Prints the dimensions of the boundary array and its content."""
        print(f"N_x_coord (includes 2 extra points at the edges) = {self.interfaces_matrix.shape[0]}, N_interfaces = {self.interfaces_matrix.shape[1]}")
        print("Interface Matrix (Transposed) \n", self.interfaces_matrix.T) #Transpose as formatted already in x,z. Numpy pretty prints for z,x

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

    def resolving_overlapped_interfaces(self, overlap_resolving_technique=None):
        """
        Post-process interfaces to remove overlaps.

        Parameters
        ----------
        overlap_resolving_technique : 'erosion' or 'reverse_erosion'
        """
        #b_line_filtered_dict, zlim, top_priority=True):
        # Process 1: Limiting the boundaries to 0 and zlim, if trim_interfaces
        # Process 2: Interface crossing handling - Currently, priority given to lower interface (v3: option to reverse the priority)
        
        if overlap_resolving_technique is None:
            overlap_resolving_technique = self._overlap_resolving_technique
        
        if overlap_resolving_technique=='erosion':    
            b_array = self.interfaces_matrix
            for i in np.arange(1, b_array.shape[-1]):
                b_array[:, i] = np.maximum(b_array[:, i], b_array[:, i-1])
        elif overlap_resolving_technique=='reverse_erosion':    
            b_array = self.interfaces_matrix[:, 1:]
            for i in np.arange(b_array.shape[-1]-2, -1, -1):
                b_array[:, i] = np.minimum(b_array[:, i], b_array[:, i+1])
        else:
            raise ValueError(f"overlap_resolving_technique must be 'erosion' or 'reverse_erosion'. Provided {overlap_resolving_technique}")
                
        # if trim_interfaces:
        #     trim_z = self.domain.spans[1]
        # else:
        #     trim_z = max(self.interfaces_matrix)
            
        # b_array=np.clip(b_array, 0, trim_z) #clip between 0 and trim_z           
        self.set_interfaces_matrix(b_array)
        self._overlap_resolving_technique = overlap_resolving_technique
    
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
        DiscretizedInterfaces2DReadOnly
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
            new._overlap_resolving_technique = self._overlap_resolving_technique
            new._adjust_top_surface_zero = self._adjust_top_surface_zero
            # Possible issue of crisscross after remeshing on some rare conditions.
            if new._locked:
                flag, min_val = new.check_if_overlapping_interfaces()
                if flag:
                    new._locked = False
                    new.resolving_overlapped_interfaces()
                    new.lock_interfaces()
                    if min_val > 1e-5:
                        warnings.warn("Overlapping interfaces found after remesh; Applied default erosion processing at the edges to correct them. This should not affect most models.")
        else:
            new.interfaces_matrix = self.interfaces_matrix.copy()
            
        # if self._locked:
        #     new.lock_interfaces()
        return new
     
    def plot(self, ax=None, plot_extents=True, legend=False, **kwargs):
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
        
        
        for i in np.arange(0, n_soil_layers):
            if i == 0:
                linestyle = '-'
                legend_label = 'Interface 0 (Surface)'
            else:
                linestyle = '-'
                legend_label = f'Interface {i}'
            ax.plot(self.domain.get_interface_x_centers,
                    self.interfaces_matrix[:, i],label=legend_label,
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

        if legend:
            ax.legend()
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

    def get_surface_and_subsurface_interfaces(self, relative_to_surface=False) -> "DiscretizedInterfaces2DReadOnly":
        """
        Seperate surface interface and return main and surface interface after seperating.
        Surface interface is the top interface.
        """
        if self.n_soil_layers == 1:
            raise ValueError("Surface interface cannot be seperated, provided the interface class has no soil-soil interfaces. Need at least one to seperate.")
        
        surf_rough = self._adj_roughness_multipliers[0]
        surface_interface_instance = DiscretizedInterfaces2DReadOnly(self.domain, 1, generate_surface=True, remesh_interp_method=self.remesh_interp_method, rng=self.rng)
        surface_interface_instance.set_interfaces_matrix(self.interfaces_matrix[:,0:1])
        self._adj_roughness_multipliers=[surf_rough]
        surface_interface_instance._adjust_top_surface_zero = self._adjust_top_surface_zero
        
        new_scale = self._adj_roughness_multipliers.copy()
        new_scale[0] = 0
        soil_interface_instance = DiscretizedInterfaces2DReadOnly(self.domain, self.n_soil_layers, generate_surface=False, remesh_interp_method=self.remesh_interp_method, rng=self.rng)
        new_interface = np.zeros_like(self.interfaces_matrix)
        
        if self.n_soil_layers > 1:
            if relative_to_surface:
                new_interface[:, 1:] = self.interfaces_matrix[:, 1:] - self.interfaces_matrix[:, :1]
            else:
                new_interface[:, 1:] = self.interfaces_matrix[:, 1:]

        soil_interface_instance.set_interfaces_matrix(new_interface)
        self._adj_roughness_multipliers=new_scale
        soil_interface_instance._overlap_resolving_technique = self._overlap_resolving_technique
        
        if self._locked:
            soil_interface_instance.lock_interfaces()
            surface_interface_instance.lock_interfaces()
            
        return soil_interface_instance, surface_interface_instance
    
    
    
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
            'adj_roughness_multipliers':self._adj_roughness_multipliers,
            'rng_state':self.rng.bit_generator.state,
            'locked':self._locked,
            'ref_x': self._ref_x,
            'overlap_resolving_technique':self._overlap_resolving_technique,
            'adjust_top_surface_zero': self._adjust_top_surface_zero
        }
        
    @classmethod
    def from_config(cls, config_dict):
        """Create 'DiscretizedInterfaces2DReadOnly' class instance from a configuration dictionary."""
        if not isinstance(config_dict, dict):
            raise TypeError("Expected a dictionary.")
        try:
            discretizedDomain2D = DiscretizedDomain2D.from_config(config_dict['domain'])
            interfaces_matrix = np.array(config_dict['interfaces_matrix'])
            n_soil_layers, generate_surface = config_dict['n_soil_layers'], config_dict['generate_surface']
            remesh_interp_method = config_dict['remesh_interp_method']

            rng = np.random.default_rng()
            rng.bit_generator.state = config_dict['rng_state']

            obj = cls(discretizedDomain2D, n_soil_layers, generate_surface, 
                       remesh_interp_method, rng)
            obj.set_interfaces_matrix(interfaces_matrix)
            obj._ref_x = config_dict['ref_x']
            obj._overlap_resolving_technique = config_dict['overlap_resolving_technique']
            obj._adj_roughness_multipliers = config_dict['adj_roughness_multipliers']
            obj._adjust_top_surface_zero = config_dict['adjust_top_surface_zero']
            
            if config_dict['locked']:
                obj.lock_interfaces()
            return obj
        
        except (KeyError, TypeError) as e:
            raise ValueError(f"Invalid config dictionary: {e}")
        
    # def __eq__(self, other, gen_params_check=False):
    #     if not isinstance(other, DiscretizedInterfaces2DReadOnly):
    #         return NotImplemented
        
    #     domain_check  = self.domain == other.domain
    #     interfaces_check = f.safe_equal(self.interfaces_matrix, other.interfaces_matrix)
    #     n_soils_check    = f.safe_equal(self.n_soil_layers, other.n_soil_layers)
    #     n_soil_if_check  = f.safe_equal(self.n_soil_soil_interfaces, other.n_soil_soil_interfaces)
    #     remesh_check     = f.safe_equal(self.remesh_interp_method, other.remesh_interp_method)
    #     rough_check      = f.safe_equal(self.roughness_multiplier, other.roughness_multiplier)
    #     ref_check        = f.safe_equal(self._ref_x, other._ref_x)
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