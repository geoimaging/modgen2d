# Used by Modules: Generating_Vs_Profile, General_Functions, Spatial_Vs
# 6/27/2023: Added discrete2continous_pdf
# 3/27/2025: Cleaned

"""Random number generator utilities."""


import matplotlib.pyplot as plt
import numpy as np
from abc import ABC, abstractmethod

class RandomGeneratorAbstract(ABC):
    """
    Abstract base class for random generators.

    All random generators must implement the ``generate`` method.
    """
    def __init__(self, rng=None):
        """
        Initializes the random generator.
        
        Parameters
        ----------
        rng : numpy.random.Generator, optional
            NumPy random number generator instance. If None, a default
            generator is created.
        """
        self.rng = rng or np.random.default_rng()

    @abstractmethod
    def generate(self, size=None):
        """
        Generate random samples.

        Parameters
        ----------
        size : int or tuple of int, optional
            Output shape. If None, a single value is returned.

        Returns
        -------
        object or numpy.ndarray
            Generated random value(s).
        """
        pass

    def plot_pmf(self, ax=None, n=1000000, bins=100):
        """
        Plot the probability mass or density function.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axis to plot on. If None, a new figure is created.
        n : int, optional
            Number of samples used for estimating the distribution.
        bins : int, optional
            Number of bins for histogram-based plots.

        Returns
        -------
        matplotlib.axes.Axes
            Axis containing the plot.
        """
        if ax is None:
            fig, ax = plt.subplots()
        
        data = self.generate(size=(n,))
        
        # Continuous distributions
        if isinstance(self, (Uniform, LogUniform, Normal)):
            ax.hist(data, bins=bins, density=True)
            ax.set_xlabel('x')
            ax.set_ylabel('Density')
            ax.set_title(f'{self.__class__.__name__} PDF')
        # Constant
        elif isinstance(self, Constant):
            ax.bar([self.value], [1])
            ax.set_xlabel('x')
            ax.set_ylabel('P(x)')
            ax.set_title('Constant PMF')
        # Discrete distributions
        # see DiscreteChoice below (as an example for overriding)
        else:
            raise NotImplementedError("Plotting not implemented for this class.")

        return ax
    
class Constant(RandomGeneratorAbstract):
    """
    Constant-valued random generator.

    Always returns the same value.
    """
    def __init__(self, val, rng=None):
        """
        Initialize a constant generator.

        Parameters
        ----------
        val : float
            Constant value to return.
        rng : numpy.random.Generator, optional
            Random generator (unused but kept for interface consistency).
        """
        super().__init__(rng)
        
        if val is None or (isinstance(val, float) and np.isnan(val)):
            raise ValueError("Value cannot be None or nan")
        
        self.value = val
    
    def generate(self, size=None):
        """
        Generate constant samples.

        Parameters
        ----------
        size : int or tuple of int, optional
            Output shape. If None, a single value is returned.

        Returns
        -------
        object or numpy.ndarray
            Constant value(s).
        """
        # If size is None, return a single value instead of an array
        single_value = size is None
        size = (1,) if single_value else size

        gen = np.full(size, self.value, dtype=object)

        return gen[0] if single_value else gen
        

class Uniform(RandomGeneratorAbstract):
    """
    Uniform random generator.
    """
    def __init__(self, low: float, high: float, rng=None):
        """
        Initialize a uniform distribution.

        Parameters
        ----------
        low : float
            Lower bound of the distribution.
        high : float
            Upper bound of the distribution.
        rng : numpy.random.Generator, optional
            Random number generator.
        """
        super().__init__(rng)
    
        if not (isinstance(low, (int,float)) or not isinstance(high, (int,float))):
            raise TypeError(f"low and high must be numbers. Provided {low} and {high}")

        if np.isnan(low) or np.isnan(high):
            raise ValueError(f"low or high cannot be nan. Provided {low} and {high}")
        
        if low>high:
            raise ValueError(f"low cannot be higher than high. Provided {low} and {high}")
        
        self.low = low
        self.high = high
    
    def generate(self, size=None):
        """
        Generate uniformly distributed samples.

        Parameters
        ----------
        size : int or tuple of int, optional
            Output shape. If None, a single value is returned.

        Returns
        -------
        object or numpy.ndarray
            Generated samples.
        """
        single_value = size is None
        size = (1,) if single_value else size

        gen = np.array(self.rng.uniform(self.low, self.high, size=size).tolist(), dtype=object)
        return gen[0] if single_value else gen
    
class LogUniform(Uniform):
    """
    Log-uniform random generator.
    """
    def __init__(self, low: float, high:float, rng=None):
        """
        Initialize a LogUniform distribution.

        Parameters
        ----------
        low : float
            Lower bound of the distribution.
        high : float
            Upper bound of the distribution.
        rng : numpy.random.Generator, optional
            Random number generator.
        """
        super().__init__(low, high, rng)
        
        if low<=0 or high<=0:
            raise ValueError("Lows and/or highs cannot be zero or negative")

    def generate(self, size=None):
        """
        Generate log-uniformly distributed samples.

        Parameters
        ----------
        size : int or tuple of int, optional
            Output shape. If None, a single value is returned.

        Returns
        -------
        object or numpy.ndarray
            Generated samples.
        """
        # If size is None, return a single value instead of an array
        single_value = size is None
        size = (1,) if single_value else size

        gen = 10 ** self.rng.uniform(np.log10(self.low), np.log10(self.high), size=size)
        gen = np.array(gen.tolist(), dtype=object)  

        return gen[0] if single_value else gen

class Normal(RandomGeneratorAbstract):
    """
    Normal (Gaussian) random generator.
    """
    def __init__(self, mean:float, stdev: float, rng=None):
        """
        Initialize a normal distribution.

        Parameters
        ----------
        mean : float
            Mean of the distribution.
        stdev : float
            Standard deviation of the distribution.
        rng : numpy.random.Generator, optional
            Random number generator.
        """
        super().__init__(rng)
        
        if not (isinstance(mean, (int,float)) and isinstance(stdev, (int,float))):
            raise TypeError("mean and stdev must be numbers")

        if np.isnan(mean) or np.isnan(stdev):
            raise ValueError("mean and stdev cannot be nan")
        
        self.mean = mean
        self.stdev = stdev

    def generate(self, size=None):
        """
        Generate normally distributed samples.

        Parameters
        ----------
        size : int or tuple of int, optional
            Output shape. If None, a single value is returned.

        Returns
        -------
        object or numpy.ndarray
            Generated samples.
        """
        # If size is None, return a single value instead of an array
        single_value = size is None
        size = (1,) if single_value else size

        gen = self.rng.normal(self.mean, self.stdev, size=size)
        gen = np.array(gen.tolist(), dtype=object)  

        return gen[0] if single_value else gen
    
class DiscreteChoice(RandomGeneratorAbstract):
    """
    Discrete random choice generator.
    """
    def __init__(self, x, p = None, rng=None):
        """
        Initialize a discrete choice distribution.

        Parameters
        ----------
        x : array-like
            Possible discrete values (all numeric or all strings).
        p : array-like, optional
            Probabilities associated with each value. Must sum to 1.
            If None, a uniform distribution is used.
        rng : numpy.random.Generator, optional
            Random number generator.
        """
        super().__init__(rng)
        
        # Convert to numpy array if not already
        x = np.asarray(x)
        p = np.asarray(p, dtype=float) if p is not None else np.ones(x.shape) / x.shape[0]
        
        if np.isnan(p).any():
            raise ValueError("p contains NaN values")

        # Ensure both x and p are 1D
        if x.ndim != 1 or p.ndim != 1:
            raise ValueError("Both x Wand p must be 1D lists or 1D numpy arrays.")

        if len(x) != len(p):
            raise TypeError(f"The length of x and p must be same. Provided {len(x)} and {len(p)}.")
        
        if np.issubdtype(x.dtype, np.number):
            pass  # all good, x is numeric
        elif all(isinstance(i, str) for i in x.tolist()):
            pass  # all good, x is string
        else:
            raise TypeError("x must contain either all numbers or all strings.")
        # Assert sum of values equal 1.
        if not np.isclose(sum(p), 1):
            raise ValueError("Sum of probabilities in diction must equal 1.")
        
        self.x = x
        self.p = p
    
    def generate(self, size=None):
        """
        Generate discrete samples.

        Parameters
        ----------
        size : int or tuple of int, optional
            Output shape. If None, a single value is returned.

        Returns
        -------
        object or numpy.ndarray
            Generated samples.
        """
        # If size is None, return a single value instead of an array
        single_value = size is None
        size = (1,) if single_value else size

        gen = self.rng.choice(self.x, size=size, p=self.p)
        gen = np.array(gen.tolist(), dtype=object)  

        return gen[0] if single_value else gen
        
    def plot_pmf(self, ax=None, n=1000000):
        """
        Plot the probability mass function.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axis to plot on. If None, a new figure is created.
        n : int, optional
            Number of samples used for estimating probabilities.

        Returns
        -------
        matplotlib.axes.Axes
            Axis containing the plot.
        """
        if ax is None:
            fig, ax = plt.subplots()
        
        data = self.generate(size=(n,))
        categories = np.unique(self.x)
        counts = [np.sum(data == cat)/len(data) for cat in categories]
        ax.bar([str(c) for c in categories], counts)
        ax.set_xlabel('x')
        ax.set_ylabel('P(x)')
        ax.set_title('Discrete PMF')
        ax.set_ylim(0, 1)
        return ax
    
class Discrete2ContinuousPDF(DiscreteChoice):
    """
    Convert a discrete probability distribution into a continuous-like PDF
    using linear interpolation.
    """
    def __init__(self, x, p, new_del_x:float, new_x_min:float=None, new_x_max:float=None, incr_x_init:float=0, rng=None):
        """
        Refine a discrete PDF into a continuous approximation by linear interpolation at each of new_x_array.

        Parameters
        ----------
        x : array-like
            Discrete support values (numeric only).
        p : array-like
            Probabilities associated with ``x`` (must sum to 1). If provided empty list; then uniform distribution over all x
        new_del_x : float
            Step size for the refined continuous grid.
        new_x_min : float, optional
            Minimum bound of the refined grid.
        new_x_max : float, optional
            Maximum bound of the refined grid.
        incr_x_init : float, optional
            Constant shift applied to ``x`` before interpolation.
        rng : numpy.random.Generator, optional
            Random number generator.
        refines discrete pdf to more of continuous one by linear interpolation at each of new_x_array
        
        Example
        -------
        x = [0,1,5]; p = [0, 0.2, 0.4] -> means p(0) = 0; p(1) = 0.2; p[5]=0.4
           
        if new_del_x = 0.5
        then, {0:0, 0.5:0.1, 1:0.2, 1.5:0.225, 2:0.25 .... ,5:0.4}. i.e. linear interpolation in between
        then, pdf_values readjusted such that sum is 1.
        """
        # Convert to numpy array if not already
        x = np.asarray(x, dtype=float)
        p = np.asarray(p, dtype=float) if p is not None else np.ones_like(x) / len(x)
        assert not np.isnan(p).any(), "p contains NaN values"
        
        # Ensure both x and p are 1D
        if x.ndim != 1 or p.ndim != 1:
            raise ValueError("Both x and p must be 1D lists or 1D numpy arrays.")

        # Ensure all elements in `x` are numbers only
        if not np.issubdtype(x.dtype, np.number):
            raise TypeError("All elements in x must be numeric (int or float).")

        # Apply shift to x
        x = x+incr_x_init

        if new_x_max is None:
            new_x_max = np.max(x)
        if new_x_min is None:
            new_x_min = np.min(x)

        # Ensure x_min ≤ min(x) and x_max ≥ max(x)
        if new_x_min > np.min(x):
            raise ValueError(f"x_min ({new_x_min}) must be less than or equal to the minimum value in shifted x ({x}).")
        if new_x_max < np.max(x):
            raise ValueError(f"x_max ({new_x_max}) must be greater than or equal to the maximum value in shifted x ({x}).")

        new_x_array = np.arange(new_x_min, new_x_max + new_del_x, new_del_x)
        new_p = np.interp(new_x_array, x, p)
        
        # Normalize probabilities so they sum to 1
        new_p = new_p / np.sum(new_p)
        
        super().__init__(x = new_x_array, p = new_p, rng=rng)