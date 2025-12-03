from math import log10, isclose

class Units:
    """
    A class to manage and validate conversion between physical length units (used by users)
    and domain length units (used internally in discretized models).

    This class ensures consistent scaling between user-specified real-world measurements
    and the internal computational domain, which operates in integer-based "domain units."

    Attributes
    ----------
    domain_length_unit : str
        The unit used internally by the computational domain (e.g., 'cm').
        Typically smaller or discretized (integer-based representation).
    physical_length_unit : str
        The physical measurement unit used by the user (e.g., 'm').
        Represents real-world scale.
    conversion_factor : int
        Conversion factor from the physical unit to the domain unit.
        Must be a power of 10 (1, 10, 100, 1000, ...).
        Example: if physical_length_unit='m' and domain_length_unit='cm', then conversion_factor=100.

    Notes
    -----
    - Users specify all model parameters and geometry in **physical length units** (e.g., meters).
    - Internally, discretized domain dimensions are converted to **domain length units** for computation,
      ensuring that domain spans and grid sizes remain integer-based.

    Examples
    --------
    >>> u = Units()
    >>> u.set_units("cm", "m", 100)
    >>> u.to_domain_length_unit(1.25)
    125
    >>> u.to_physical(250)
    2.5
    """

    def __init__(self):
        """Initialize with default units (domain: 'cm', physical: 'm', conversion: 100)."""
        self.domain_length_unit = "cm"
        self.physical_length_unit = "m"
        self.conversion_factor = 100

    def set_units(self, domain_length_unit: str, physical_length_unit: str, conversion_factor: int):
        """
        Set and validate unit configuration.

        Parameters
        ----------
        domain_length_unit : str
            Unit used internally in the computational domain (e.g., 'cm').
            Must be ≤ 4 characters.
        physical_length_unit : str
            Real-world measurement unit (e.g., 'm').
            Must be ≤ 4 characters.
        conversion_factor : int or float
            Conversion factor from physical_length_unit to domain_length_unit.
            Must be 10^n (1, 10, 100, 1000, ...).
        """
        # --- Validate string units ---
        for name, val in {
            "domain_length_unit": domain_length_unit,
            "physical_length_unit": physical_length_unit,
        }.items():
            if not isinstance(val, str):
                raise TypeError(f"{name} must be a string.")
            if len(val) > 4:
                raise ValueError(f"{name} ('{val}') must have ≤ 4 characters.")

        # --- Validate conversion factor ---
        if not isinstance(conversion_factor, (int, float)):
            raise TypeError("conversion_factor must be numeric.")
        if conversion_factor <= 0:
            raise ValueError("conversion_factor must be positive.")

        log_val = log10(conversion_factor)
        if not isclose(log_val, round(log_val)):
            raise ValueError(
                f"conversion_factor ({conversion_factor}) must be 10^n where n is a whole number."
            )

        # --- Assign values ---
        self.domain_length_unit = domain_length_unit
        self.physical_length_unit = physical_length_unit
        self.conversion_factor = int(conversion_factor)

    def to_domain_length_unit(self, physical_value: float) -> int:
        """
        Convert a physical value to domain units.

        The resulting value must be an exact integer after conversion;
        otherwise, an error is raised.

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
            return ValueError(f"Lengths cannot be negative. Provided {physical_value}")

        converted = physical_value * self.conversion_factor
        if not isclose(converted, round(converted), abs_tol=0):
            raise ValueError(
                f"Converted domain value ({converted}) is not an integer. "
                f"Ensure the physical value aligns with the discretized grid."
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
        return domain_value / self.conversion_factor
    
    def __eq__(self, other):
        if not isinstance(other, Units):
            return NotImplemented
        return (
            self.domain_length_unit == other.domain_length_unit
            and self.physical_length_unit == other.physical_length_unit
            and self.conversion_factor == other.conversion_factor
        )
        
    @property
    def get_config(self):
        return {
            'domain_length_unit': self.domain_length_unit,
            'physical_length_unit': self.physical_length_unit,
            'conversion_factor':self.conversion_factor,
        }
    
    @classmethod
    def from_config(cls, config_dict):
        if not isinstance(config_dict, dict):
            raise TypeError("Expected a dictionary.")
        try:
            dlu = config_dict['domain_length_unit']
            plu = config_dict['physical_length_unit']
            cf = config_dict['conversion_factor']
            return cls().set_units(dlu, plu, cf)
        except (KeyError, TypeError) as e:
            raise ValueError(f"Invalid config dictionary: {e}")
