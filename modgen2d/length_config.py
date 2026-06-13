"""Units configuration for discretized domains."""

from math import log10, isclose

class LengthConfig:
    """Manage the physical length unit and minimum spatial resolution
    used by the computational model.
    
    Parameters
    ----------
    physical_length_unit : str, optional
        The physical measurement unit used by the user (e.g., 'm').
        Represents real-world scale, and one to be used externally throughout.
    min_dl: float, optional
        Minimum spatial increment (dl) allowed by the model for stable remeshing.
        It's 'inverse' must be 10^N where N is a whole number. 
        All dl values must be multiples of this value. 
        Can be set to None if max_grid_density is provided.
    max_grid_density : int, optional
        If min_dl is None, then max_grid_density is used. 
        If both provided, performs additional check if they are consistent (Must be inverse of min_dl.) 
        Used as a conversion factor from the external 'physical' unit to the internal 'domain' unit.

    Raises
    ------
    TypeError
        If inputs have invalid types.
    ValueError
        If inputs are invalid.
    
    Notes
    -----
    - Users specify all model parameters and geometry in **length units** (e.g., meters).
    - Users must provide either min_dl or max_grid_density
    - Internally, discretized domain dimensions are internally converted to **domain length units** for computation,
      ensuring that domain spans and grid sizes remain integer-based.

    Examples
    --------
    >>> u = LengthConfig("m", 0.01)
    >>> u.to_domain_length_unit(1.25)
    125
    >>> u.to_physical_length_unit(250)
    2.5
    """
    
    def __init__(
        self,
        physical_length_unit: str = "m",
        min_dl: float = None,
        max_grid_density: int = None,
    ):
        """
        Initialize a ModelResolution class object. 
        """
        self.__set_model_resolutions(physical_length_unit, min_dl, max_grid_density)

    def __set_model_resolutions(self, physical_length_unit: str, min_dl: float, max_grid_density:int):
        """
        Set and validate unit configuration.

        Parameters
        ----------
        physical_length_unit : str
            Real-world measurement unit (e.g., 'm').
            Must be ≤ 4 characters.
        min_dl: float, optional
            Minimum spatial increment (dl) allowed by the model for stable remeshing.
            All dl values must be multiples of this value.
        """
        # --- Validate string units ---
        if not isinstance(physical_length_unit, str):
            raise TypeError(f"physical_length_unit must be a string.")
        if len(physical_length_unit) > 4:
            raise ValueError(f"physical_length_unit ('{physical_length_unit}') must have ≤ 4 characters.")

        # --- Validate min_dl ---
        if min_dl is not None:
            min_dl = float(min_dl)
            if not isinstance(min_dl, (int, float)):
                raise TypeError("min_dl must be numeric.")
            if min_dl <= 0:
                raise ValueError("min_dl must be positive.")

        else:
            if max_grid_density is None:
                raise ValueError("Both min_dl and max_grid_density cannot be None.")
            min_dl = 1/max_grid_density
                    
        if max_grid_density is not None:
            max_grid_density = float(max_grid_density)
            if not isinstance(max_grid_density, (int, float)):
                raise TypeError("conversion_factor must be numeric.")
            if max_grid_density <= 0:
                raise ValueError("conversion_factor must be positive.")
            
            if not isclose(max_grid_density, 1/min_dl):
                raise ValueError(
                    f"max_grid_density = {max_grid_density} and min_dl {min_dl} must be inverse to each other."
                )
        else:
            max_grid_density = 1/min_dl
                   
        log_val = log10(max_grid_density)
        if not isclose(log_val, round(log_val)):
            raise ValueError(
                f"max_grid_density (1/min_dl) = {max_grid_density} must be 10^n where n is a whole number. Not the case for provided min_dl ({min_dl})."
            )
        max_grid_density = 10**round(log_val)
        # Assign values bypassing immutability 
        object.__setattr__(self, "physical_length_unit", physical_length_unit)
        object.__setattr__(self, "min_dl", 1/max_grid_density)
        object.__setattr__(self, "max_grid_density", max_grid_density)

    def __setattr__(self, name, value):
        """Prevent modification of attributes."""
        raise AttributeError(
            f"{self.__class__.__name__} instances are immutable. "
            "Create a new instance to change values."
        )

    def to_domain_length_unit(self, physical_value: float) -> int:
        """
        Convert a physical value to domain units.

        All discretized domain uses this method for conversion.
        
        Parameters
        ----------
        physical_value : float
            Value in physical units (e.g., meters).

        Returns
        -------
        int
            Value in domain units (integer-based).

        Raises
        ------
        TypeError
            If the input is not numeric.
        ValueError
            If the converted value is not an integer.
        """
        if not isinstance(physical_value, (int, float)):
            raise TypeError("physical_value must be numeric.")

        if physical_value < 0:
            raise ValueError(f"Lengths cannot be negative. Provided {physical_value}")

        converted = physical_value * self.max_grid_density
        if not isclose(converted, round(converted), abs_tol=0):
            raise ValueError(
                f"Converted domain value ({converted}) is not an integer. "
                f"Ensure the physical value is a integer in domain_units."
            )

        return int(round(converted))

    def to_physical_length_unit(self, domain_value: int) -> float:
        """
        Convert a domain unit value back to physical units.

        Parameters
        ----------
        domain_value : int
            Value in domain units (integer-based).

        Returns
        -------
        float
            Value in physical units (e.g., meters).
        """
        if not isinstance(domain_value, (int, float)):
            raise TypeError("domain_value must be numeric.")
        return domain_value / self.max_grid_density
    
    def __eq__(self, other):
        if not isinstance(other, LengthConfig):
            return NotImplemented
        return (
            self.physical_length_unit == other.physical_length_unit
            and self.min_dl == other.min_dl
            and self.max_grid_density == other.max_grid_density
        )
        
    @property
    def get_config(self):
        """Return unit configuration as a dictionary."""
        return {
            'physical_length_unit': self.physical_length_unit,
            'min_dl': self.min_dl,
            'max_grid_density':self.max_grid_density,
        }
    
    @classmethod
    def from_config(cls, config_dict):
        """Create Units class instance from a configuration dictionary."""
        if not isinstance(config_dict, dict):
            raise TypeError("Expected a dictionary.")
        try:
            plu = config_dict['physical_length_unit']
            min_dl = config_dict['min_dl']
            cf = config_dict['max_grid_density']
            return cls(plu, min_dl, cf)
        except (KeyError, TypeError) as e:
            raise ValueError(f"Invalid config dictionary: {e}")
