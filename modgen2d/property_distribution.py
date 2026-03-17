# Sanish Bhochhibhoya
# Apr 12: Changed the Structure of properties. 
#   Was: Main Property >> Feature ID >> wet/dry/both >> Dict of Random Generators or Random Generator
#   Now: Main Property >> Feature ID >> Dictionary of material types as keys >> Wet/dry/both >> Random Generator Instance.

from modgen2d.random_generators import RandomGeneratorAbstract

class PropertyDistribution:
    __slots__ = ['_property_name', '_mean_distribution', '_stdev_distribution', '_stdev_type', '_description', '_check']
    """
        Defines the probabilistic distribution of a material property.
        A property is described by a mean distribution and an optional standard deviation (or coefficient of variation) distribution.
    """
    def __init__(self, property_name, mean_distribution, stdev_distribution=None, stdev_type = 'stdev', mean_slope_with_depth_distribution = None, description = ''):
        """
        Initializes the 'PropertyDistribution' object.

        Parameters
        ----------
        property_name : str
            Name of the property (must match the parent MainProperty name).
        mean_distribution : RandomGeneratorAbstract
            Random generator defining the mean value.
        stdev_distribution : RandomGeneratorAbstract or None, optional
            Random generator defining the standard deviation or coefficient
            of variation. If None, zero variance is assumed.
        stdev_type : {'stdev', 'cov'}, optional
            Type of variability definition. If 'cov', standard deviation is
            computed as mean × cov.
        mean_slope_with_depth_distribution : RandomGeneratorAbstract
            Random generator defining the slope of mean value (w/ depth). None, if not provided!
        description : str, optional
            Human-readable description of conditions or assumptions.
        """
        self._property_name = property_name
        self._check_distribution(mean_distribution)
        self._mean_distribution = mean_distribution
        
        if stdev_type not in ['stdev', 'cov']:
            raise ValueError("stdev_type must be either 'stdev', or 'cov'.")
        
        self._stdev_type = stdev_type
        if stdev_distribution is None:
            self._stdev_distribution = None 
        else:
            self._check_distribution(stdev_distribution)
            self._stdev_distribution = stdev_distribution 
            
        if mean_slope_with_depth_distribution is None:
            self._mean_slope_with_depth_distribution = None
        else:
            self._check_distribution(mean_slope_with_depth_distribution)
            self._mean_slope_with_depth_distribution = mean_slope_with_depth_distribution 

        self._description = description # Condition details: Eg. If mean or variance is dependent to anything, if so will have a user-understandable conditions mentioned like "soil type", "Vs > "
        self._check = False # Validation check flag

    def _check_distribution(self, distribution_to_check):
        """
        Validates if the given distribution is correctly formatted.

        Parameters
        ----------
        distribution_to_check : RandomGeneratorAbstract
            Distribution to be validated.

        Raises
        ------
        TypeError
            If the distribution is not a RandomGeneratorAbstract instance.
        """
        
        if not isinstance(distribution_to_check, RandomGeneratorAbstract):
            raise TypeError(f"set_dist must be an instance of a subclass of a RandomGenerator class.")
            
    def check_class(self):
        """
        Performs internal validation on the distribution assignments.

        Checks:
        1) Mean distribution must not be 'NA'; like when initialized.
        2) Both Std and cov distribution must not be 'NA'; but both cannot be defined too. 
        3) The distributions (mean, or one of std/cov) is correctly defined.
        """
        self._check = False
        
        self._check_distribution(self._mean_distribution)
        if self._stdev_distribution is not None:
            self._check_distribution(self._stdev_distribution)
            
        if self._mean_slope_with_depth_distribution is not None:
            self._check_distribution(self._mean_slope_with_depth_distribution)
            
        self._check = True
            
    @property
    def property_name(self):
        return self._property_name
    
    @property
    def mean_distribution(self):
        return self._mean_distribution

    @property
    def stdev_distribution(self):
        return self._stdev_distribution

    @property
    def stdev_type(self):
        return self._stdev_type

    @property
    def mean_slope_with_depth_distribution(self):
        return self._mean_slope_with_depth_distribution
    
    @property
    def description(self):
        return self._description

    @property
    def check(self):
        return self._check
      