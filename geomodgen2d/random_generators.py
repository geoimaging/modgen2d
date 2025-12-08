# Used by Modules: Generating_Vs_Profile, General_Functions, Spatial_Vs
# 6/27/2023: Added discrete2continous_pdf
# 3/27/2025: Cleaned

import matplotlib.pyplot as plt
import numpy as np
from abc import ABC, abstractmethod

class RandomGeneratorAbstract(ABC):
    def __init__(self, rng=None):
        """
        Initializes the random generator.
        
        Parameters:
        rng: Universal random number generator defined by np.random.default_rng(seed=r_seed)
        """
        self.rng = rng or np.random.default_rng()

    @abstractmethod
    def generate(self, size=(1,)):
        """Return generated data"""
        pass

    def plot_pmf(self, ax=None, n=1000000, bins=100):
        """
        Default plotting function. Subclasses can override.
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
    Random generator, but returns constant value only.
    
    Parameters:
    val (float): The constant value to return
    """
    def __init__(self, val, rng=None):
        super().__init__(rng)
        
        if val is None or (isinstance(val, float) and np.isnan(val)):
            raise ValueError("Value cannot be None or nan")
        
        self.value = val
    
    def generate(self, size=(1,)):
        return np.full(size, self.value)

class Uniform(RandomGeneratorAbstract):
    """
    Initializes the uniform random generator with a specified range.
    
    Parameters:
    low (float): The lower bound of the range.
    high (float): The upper bound of the range.
    """
    
    def __init__(self, low: float, high: float, rng=None):
        super().__init__(rng)
    
        if not (isinstance(low, (int,float)) and isinstance(high, (int,float))):
            raise TypeError("low and high must be numbers")

        if np.isnan(low) or np.isnan(high):
            raise ValueError("low or high cannot be nan")
        
        if low>high:
            raise ValueError("low cannot be higher than high.")
        
        self.low = low
        self.high = high
    
    def generate(self, size=(1,)):
        arr = self.rng.uniform(self.low, self.high, size=size)
        return np.array(arr.tolist(), dtype=object)     
    
class LogUniform(Uniform):
    """
    Initializes the log-uniform random generator with a specified range.
    
    Parameters:
    low (float): The lower bound of the range.
    high (float): The upper bound of the range.
    """
    def __init__(self, low: float, high:float, rng=None):
        super().__init__(low, high, rng)
        
        if low<=0 or high<=0:
            raise ValueError("Lows and/or highs cannot be zero or negative")

    def generate(self, size=(1,)):
        arr = 10 ** self.rng.uniform(np.log10(self.low), np.log10(self.high), size=size)
        return np.array(arr.tolist(), dtype=object)     

class Normal(RandomGeneratorAbstract):
    """
    Initializes the uniform random generator with a specified range.
    
    Parameters:
    mean (float): The mean of the distribution.
    stdev (float): The standard deviation of the distribution.
    """
    def __init__(self, mean:float, stdev: float, rng=None):
        super().__init__(rng)
        
        if not (isinstance(mean, (int,float)) and isinstance(stdev, (int,float))):
            raise TypeError("mean and stdev must be numbers")

        if np.isnan(mean) or np.isnan(stdev):
            raise ValueError("mean and stdev cannot be nan")
        
        self.mean = mean
        self.stdev = stdev

    def generate(self, size=(1,)):
        arr = self.rng.normal(self.mean, self.stdev, size=size)
        return np.array(arr.tolist(), dtype=object)     

class DiscreteChoice(RandomGeneratorAbstract):

    """
    Perform random sampling from the given values `x` with probabilities `p`.
    
    Parameters:
    x (1D-list or 1D-array): The possible choices. Either all numbers or all strings.
    p (1D-list or 1D-array): Corresponding probabilities for each choice (must sum to 1). If provided empty list; then uniform distribution over all x
    """
    def __init__(self, x, p = None, rng=None):
        super().__init__(rng)
        
        # Convert to numpy array if not already
        x = np.asarray(x, dtype=object)
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
    
    def generate(self, size=(1,)):
        arr = self.rng.choice(self.x, size=size, p=self.p)
        return np.array(arr.tolist(), dtype=object)     
        
    def plot_pmf(self, ax=None, n=1000000):
        """
        Default plotting function. Subclasses can override.
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
    def __init__(self, x, p, new_del_x:float, new_x_min:float=None, new_x_max:float=None, incr_x_init:float=0, rng=None):
        """
        refines discrete pdf to more of continuous one by linear interpolation at each of new_x_array
        
        Parameters:
            x (1D-list or 1D-array): The possible choices. Must be all numbers
            p (1D-list or 1D-array): Corresponding probabilities for each choice (must sum to 1). If provided empty list; then uniform distribution over all x
            
            x = [0,1,5]; p = [0, 0.2, 0.4] -> means p(0) = 0; p(1) = 0.2; p[5]=0.4
            incr_x_init: Value added to each element in `x` (useful for shifting).
            Use of incr_x_init in this code: we have (depth2top vs pdf) but we need to convert it to depth2center vs pdf, so if we use incr_x_init=radius of utility then it converts to what we need,
        
            new_del_x: Step size of new_x for continous pdf
            new_x_min: Minimum bound for new_x for continuous pdf (must be >= minimum of shifted_x). Default: min(x)
            new_x_max: Maximum bound of new_x for continuous pdf (must be <= minimum of shifted_x). Default: max(x)
           
        Example: if new_del_x = 0.5
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