

"""Discretized 2D computational domain utilities"""

import numpy as np
from .length_config import LengthConfig
import matplotlib.pyplot as plt
from .general_functions import is_integer_value

class DiscretizedDomain2D():
    """
    Two-dimensional spatial domain discretized into uniform elements.

    The domain is defined along x- and z-directions, with centers computed
    for each element.
    """
    
    def __init__(self, span_x: float, span_z: float, dx: float, dz: float, length_config:LengthConfig = None, origin_x=0):
        """
        Initialize a DiscretizedDomain2D class object.
        
        Parameters
        ----------
        span_x, span_z : float
            Domain size in the x-direction and z-direction respectively (physical units).
        dx, dz : float
            Discretization step in x-direction ans z-direction (physical units).
        length_config : LengthConfig, optional
            LengthConfig configuration for conversion. Defaults to ``LengthConfig()`` -> physical units is "m", and min_dl = 0.0001.
        origin_x : float
            Origin for x Note: (end_x = origin_x + span_x). For z: Origin is always zero.
        Raises
        ------
        ValueError
            If the discretization is invalid (spans not divisible by steps).
        """
        span_z = float(span_z)
        dz = float(dz)
        
        # if span_x is not None and span_x != 0:
        origin_x = float(origin_x)
        origin_z = 0
        
        span_x = float(span_x)
        dx = float(dx)
            # self._dim = 2 #2d
        # else:
        #     span_x = span_z/10
        #     dx = span_x  # Making sure there is only one point in x axis.
        #     self._dim = 1 #2d
        
        if length_config is None:
            length_config = LengthConfig("m", max_grid_density=100) # Default config - cm, m
        
        self.length_config = length_config
        
        self._spans_in_domain_len_units = [length_config.to_domain_length_unit(span_x),
                                           length_config.to_domain_length_unit(span_z)]
        
        self._origins_in_domain_len_units = [length_config.to_domain_length_unit(origin_x),
                                           length_config.to_domain_length_unit(origin_z)]
        
        self._dhs_in_domain_len_units = [length_config.to_domain_length_unit(dx), 
                                         length_config.to_domain_length_unit(dz)]
        
        # check discretization is valid
        if not self.is_valid_mesh(self._spans_in_domain_len_units, self._dhs_in_domain_len_units, self._origins_in_domain_len_units):
            msg = f"Requirements: origins, spans and dels must be positive integers in domain units; and spans must be divisible by respective dhs. Note: {length_config.physical_length_unit} = {length_config.max_grid_density} domain_units. Provided:- spans in domain units = {self._spans_in_domain_len_units}; dhs in domain_units={self._dhs_in_domain_len_units}; origins in domain_units = {self._origins_in_domain_len_units}."
            raise ValueError(msg)
        
        # define centers of each element
        self.x_centers = (self._origins_in_domain_len_units[0] + (np.arange(0, self._spans_in_domain_len_units[0], self._dhs_in_domain_len_units[0]) + self._dhs_in_domain_len_units[0] / 2))/length_config.max_grid_density
        self.z_centers = (self._origins_in_domain_len_units[1] + (np.arange(0, self._spans_in_domain_len_units[1], self._dhs_in_domain_len_units[1]) + self._dhs_in_domain_len_units[1] / 2))/length_config.max_grid_density
        self.shape = (len(self.x_centers), len(self.z_centers))
        
    @staticmethod
    def is_valid_discretization(span_domain_unit, delta_domain_unit, origin_domain_unit):
        """ Check if a span is divisible by a discretization step."""
        integer_check = is_integer_value(span_domain_unit) and is_integer_value(delta_domain_unit) and is_integer_value(origin_domain_unit)
        return integer_check and np.isclose(span_domain_unit % delta_domain_unit, 0)
        
    @staticmethod
    def is_valid_mesh(spans_domain_unit, dhs_domain_unit, origins_domain_unit):
        """Validate mesh spans and discretization steps."""
        try:
            for span, dh, origin in zip(spans_domain_unit, dhs_domain_unit, origins_domain_unit):
                if span <= 0:
                    return False
                if dh <= 0:
                    return False
                if not DiscretizedDomain2D.is_valid_discretization(span, dh, origin):
                    return False
            return True
        except:
            return False
        
    def can_domain_be_remeshed(self, new_dx: float, new_dz: float):
        """Check if the domain can be remeshed."""
        new_dhs_in_domain_len_units = [
            self.length_config.to_domain_length_unit(new_dx), 
            self.length_config.to_domain_length_unit(new_dz)
            ]
        return self.is_valid_mesh(self._spans_in_domain_len_units, new_dhs_in_domain_len_units, self._origins_in_domain_len_units)

    def remesh(self, new_dx:float = None, new_dz:float = None):#, inplace=True):
        """
        Return a new remeshed domain.

        Parameters
        ----------
        new_dx : float, optional
            New x discretization. Defaults to current dx.
        new_dz : float, optional
            New z discretization. Defaults to current dz.

        Returns
        -------
        DiscretizedDomain2D
            Remeshed domain.
        """
        
        if new_dx is None:
            new_dx = self.dhs[0]
            
        if new_dz is None:
            new_dz = self.dhs[1]
        
        # if inplace:
            # self.__init__(self.spans[0], self.spans[1], new_dx, new_dz, self.length_config)
        
        return DiscretizedDomain2D(self.spans[0], self.spans[1], new_dx, new_dz, self.length_config, self.origins[0])
    
    @staticmethod
    def get_minimum_domain(domain2d_list:list):
        """Return a domain with minimum discretization. """
        first = True
        for lit_domain in domain2d_list:
            if not isinstance(lit_domain, DiscretizedDomain2D):
                raise TypeError(
                    "Entries of domain2d_list must be instances of "
                    "DiscretizedDomain2D."
                )
            
            if first:
                base_domain = lit_domain 
                min_dh_x = lit_domain.dhs[0] 
                min_dh_z = lit_domain.dhs[1] 
                origin_x = lit_domain._origins_in_domain_len_units[0]
                first = False
            else:
                if not base_domain.is_equivalent(lit_domain):
                    raise ("Not all domains have same spans, origins and/or unit config.")        
                min_dh_x = min(min_dh_x, lit_domain.dhs[0])
                min_dh_z = min(min_dh_x, lit_domain.dhs[1])
                
        return DiscretizedDomain2D(base_domain.spans[0], base_domain.spans[1],
                                   min_dh_x, min_dh_z, base_domain.length_config, origin_x)
        
    def is_equivalent(self, other):
        """Check span and unit equivalence."""
        if not isinstance(other, DiscretizedDomain2D):
            raise ValueError("Other is not discretized domain")
        
        if self.length_config != other.length_config:
            return False
        
        origins_check = np.allclose(self._origins_in_domain_len_units, other._origins_in_domain_len_units)
        spans_check = np.allclose(self._spans_in_domain_len_units, other._spans_in_domain_len_units)
        return spans_check and origins_check
    
    def __eq__(self, other):
        """Check full domain equality."""
        if not isinstance(other, DiscretizedDomain2D):
            return NotImplemented
        
        units_check = self.length_config == other.length_config
        spans_check = np.allclose(self._spans_in_domain_len_units, other._spans_in_domain_len_units)
        dhs_check = np.allclose(self._dhs_in_domain_len_units, other._dhs_in_domain_len_units)
        origins_check = np.allclose(self._origins_in_domain_len_units, other._origins_in_domain_len_units)
        return (
            units_check
            and spans_check
            and dhs_check
            and origins_check
        )
    
    def plot(self, ax=None, discrete_point_size=1, edges_size=1, edges_color='b'):
        """
        Plots a 2D property profile.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Matplotlib axes to plot on.
        discrete_point_size : float, default 0
            Size of scatter points.

        Returns
        -------
        ax : matplotlib.axes.Axes
            Axes containing the plot.
        """
        if ax is None:
            fig,ax = plt.subplots()

        x_centers, z_centers = self.x_centers, self.z_centers
        x_edges, z_edges = self.x_edges, self.z_edges
        
        x_data, z_data = np.meshgrid(x_centers, z_centers, indexing='ij')
        if discrete_point_size!=0:
            ax.scatter(x_data.flatten(), z_data.flatten(), c = 'k', s=discrete_point_size)
            
        if edges_size != 0:
            for e in x_edges:
                ax.axvline(e, color=edges_color, linewidth=edges_size)
        
            for e in z_edges:
                ax.axhline(e, color=edges_color, linewidth=edges_size)
                
        ax.axis('scaled')
        ax.set(xlim = [x_edges[0], x_edges[-1]], ylim = [z_edges[0], z_edges[-1]])

        return ax    
        
    @property
    def spans(self):
        return [i/self.length_config.max_grid_density for i in self._spans_in_domain_len_units]

    @property
    def dhs(self):
        return [i/self.length_config.max_grid_density for i in self._dhs_in_domain_len_units]
    
    @property
    def origins(self):
        return [i/self.length_config.max_grid_density for i in self._origins_in_domain_len_units]
    
    @property
    def x_edges(self):
        return self.origins[0] + np.arange(0, self.shape[0] + 1) * self.dhs[0]
    
    @property
    def z_edges(self):
        return self.origins[0] + np.arange(0, self.shape[1] + 1) * self.dhs[1]
    
    @property
    def get_interface_x_centers(self):
        dx = self.dhs[0]
        return np.concatenate([ ## 2 extra points for interfaces (Useful for remeshing, no criss crossing after remeshing).
            np.array([self.x_centers[0] - dx]),
            self.x_centers,
            np.array([self.x_centers[-1] + dx])
        ])
        
    @property
    def interface_shape(self):
        return (self.shape[0]+2, self.shape[1])
        
    @property
    def get_config(self):
        """
        Return domain configuration.

        Returns
        -------
        dict
            Dictionary with spans, discretizations, and units configuration.
        """
        return {
            'spans_xz': self.spans,
            'dhs_xz': self.dhs,
            'origins_xz': self.origins,
            'length_config':self.length_config.get_config
        }
    
    @classmethod
    def from_config(cls, config_dict):
        """
        Create domain from configuration dictionary.

        Parameters
        ----------
        config_dict : dict
            Configuration dictionary.

        Returns
        -------
        DiscretizedDomain2D
            Initialized domain.
        """
        if not isinstance(config_dict, dict):
            raise TypeError("Expected a dictionary.")
        try:
            spans = config_dict['spans_xz']
            dhs = config_dict['dhs_xz']
            length_config = LengthConfig.from_config(config_dict['length_config'])
            origins = config_dict['origins_xz']
            return cls(*spans, *dhs, length_config, origin_x = origins[0])
        except (KeyError, TypeError) as e:
            raise ValueError(f"Invalid config dictionary: {e}")