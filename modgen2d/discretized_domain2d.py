

"""Discretized 2D computational domain utilities. Copied from Modgen2d"""

import numpy as np
from .units_config import Units

class DiscretizedDomain2D():
    """
    Two-dimensional spatial domain discretized into uniform elements.

    The domain is defined along x- and z-directions, with centers computed
    for each element.
    """
    
    def __init__(self, span_x: float, span_z: float, dx: float, dz: float, units_config = None):
        """
        Initialize a DiscretizedDomain2D class object.
        
         Parameters
        ----------
        span_x, span_z : float
            Domain size in the x-direction and z-direction respectively (physical units).
        dx, dz : float
            Discretization step in x-direction ans z-direction (physical units).
        units_config : Units, optional
            Unit configuration for conversion. Defaults to ``Units()`` -> physical units is "m", and domain units is "cm".

        Raises
        ------
        ValueError
            If the discretization is invalid (spans not divisible by steps).
        """
        span_z = float(span_z)
        dz = float(dz)
        
        # if span_x is not None and span_x != 0:
        span_x = float(span_x)
        dx = float(dx)
            # self._dim = 2 #2d
        # else:
        #     span_x = span_z/10
        #     dx = span_x  # Making sure there is only one point in x axis.
        #     self._dim = 1 #2d
        
        if units_config is None:
            units_config = Units() # Default config - cm, m
        
        self.units_config = units_config
        
        self._spans_in_domain_len_units = [units_config.to_domain_length_unit(span_x),
                                           units_config.to_domain_length_unit(span_z)]
        
        self._dhs_in_domain_len_units = [units_config.to_domain_length_unit(dx), 
                                         units_config.to_domain_length_unit(dz)]
        
        # check discretization is valid
        if not self.is_valid_mesh(self._spans_in_domain_len_units, self._dhs_in_domain_len_units):
            msg = f"Requirements: spans and dels must be positive and spans in {units_config.domain_length_unit}={self._spans_in_domain_len_units} must be divisible by dhs in {units_config.domain_length_unit}={self._dhs_in_domain_len_units}."
            raise ValueError(msg)
        
        # define centers of each element
        self.x_centers = (np.arange(0, self._spans_in_domain_len_units[0], self._dhs_in_domain_len_units[0]) + self._dhs_in_domain_len_units[0] / 2)/units_config.conversion_factor
        self.z_centers = (np.arange(0, self._spans_in_domain_len_units[1], self._dhs_in_domain_len_units[1]) + self._dhs_in_domain_len_units[1] / 2)/units_config.conversion_factor
        self.shape = (len(self.x_centers), len(self.z_centers))
        
    @staticmethod
    def is_valid_discretization(span, delta):
        """ Check if a span is divisible by a discretization step."""
        return np.isclose(span % delta, 0)
        
    @staticmethod
    def is_valid_mesh(spans, dhs):
        """Validate mesh spans and discretization steps."""
        try:
            for span, dh in zip(spans, dhs):
                if span <= 0:
                    return False
                if dh <= 0:
                    return False
                if not DiscretizedDomain2D.is_valid_discretization(span, dh):
                    return False
            return True
        except:
            return False
        
    def can_domain_be_remeshed(self, new_dx: float, new_dz: float):
        """Check if the domain can be remeshed."""
        new_dhs_in_domain_len_units = [
            self.units_config.to_domain_length_unit(new_dx), 
            self.units_config.to_domain_length_unit(new_dz)
            ]
        return self.is_valid_mesh(self._spans_in_domain_len_units, new_dhs_in_domain_len_units)

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
            # self.__init__(self.spans[0], self.spans[1], new_dx, new_dz, self.units_config)
        
        return DiscretizedDomain2D(self.spans[0], self.spans[1], new_dx, new_dz, self.units_config)
    
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
                first = False
            else:
                if not base_domain.is_equivalent(lit_domain):
                    raise ("Not all domains have same spans and/or unit config.")        
                min_dh_x = min(min_dh_x, lit_domain.dhs[0])
                min_dh_z = min(min_dh_x, lit_domain.dhs[1])
        
        return DiscretizedDomain2D(base_domain.spans[0], base_domain.spans[1],
                                   min_dh_x, min_dh_z, base_domain.units_config)
        
    def is_equivalent(self, other):
        """Check span and unit equivalence."""
        if not isinstance(other, DiscretizedDomain2D):
            raise ValueError("Other is not discretized domain")
        
        if self.units_config != other.units_config:
            return False
            
        spans_check = np.allclose(self._spans_in_domain_len_units, other._spans_in_domain_len_units)
        return spans_check     
    
    def __eq__(self, other):
        """Check full domain equality."""
        if not isinstance(other, DiscretizedDomain2D):
            return NotImplemented
        
        units_check = self.units_config == other.units_config
        spans_check = np.allclose(self._spans_in_domain_len_units, other._spans_in_domain_len_units)
        dhs_check = np.allclose(self._dhs_in_domain_len_units, other._dhs_in_domain_len_units)
        return (
            units_check
            and spans_check
            and dhs_check
        )
          
    @property
    def spans(self):
        return [i/self.units_config.conversion_factor for i in self._spans_in_domain_len_units]

    @property
    def dhs(self):
        return [i/self.units_config.conversion_factor for i in self._dhs_in_domain_len_units]
    
    @property
    def x_edges(self):
        return np.arange(0, self.shape[0] + 1) * self.dhs[0]
    
    @property
    def z_edges(self):
        return np.arange(0, self.shape[1] + 1) * self.dhs[1]
    
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
            'units_config':self.units_config.get_config
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
            units_config = Units.from_config(config_dict['units_config'])
            return cls(*spans, *dhs, units_config)
        except (KeyError, TypeError) as e:
            raise ValueError(f"Invalid config dictionary: {e}")