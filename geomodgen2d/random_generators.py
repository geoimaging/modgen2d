# Used by Modules: Generating_Vs_Profile, General_Functions, Spatial_Vs
# 6/27/2023: Added discrete2continous_pdf
# 3/27/2025: Cleaned

import matplotlib.pyplot as plt
import numpy as np

class RandomGeneratorDefine:
    def __init__(self, random_seed_generator=np.random.default_rng()):
        """
        Initializes the random generator.
        
        Parameters:
        random_seed_generator: Universal random number generator defined by np.random.default_rng(seed=r_seed)
        """
        self.random_seed_generator = random_seed_generator
        self.rg_type = 'NA'
        
    def constant(self, val: float):
        """
        Random generator, but returns constant value only.
        
        Parameters:
        val (float): The constant value to return
        """
        assert not np.isnan(val), "Value cannot be nan"
        self.x = np.array([val])
        self.rg_type = 'constant'
        return self  # For chaining

    def uniform(self, low: float, high: float):
        """
        Initializes the uniform random generator with a specified range.
        
        Parameters:
        low (float): The lower bound of the range.
        high (float): The upper bound of the range.
        """
        assert not np.isnan(low), "Value cannot be nan"
        assert not np.isnan(high), "Value cannot be nan"
        self.low = low
        self.high = high
        self.rg_type = 'uniform'
        return self   # For chaining

    def normal(self, mean:float, std: float):
        """
        Initializes the uniform random generator with a specified range.
        
        Parameters:
        mean (float): The mean of the distribution.
        std (float): The standard deviation of the distribution.
        """
        assert not np.isnan(mean), "Value cannot be nan"
        assert not np.isnan(std), "Value cannot be nan"
        
        self.mean = mean
        self.std = std
        self.rg_type = 'normal'
        return self   # For chaining

    def discrete_choice(self, x, p = []):
        """
        Perform random sampling from the given values `x` with probabilities `p`.
        
        Parameters:
        x (1D-list or 1D-array): The possible choices. Either all numbers or all strings.
        p (1D-list or 1D-array): Corresponding probabilities for each choice (must sum to 1). If provided empty list; then uniform distribution over all x
        """
        
        # Convert to numpy array if not already
        x = np.asarray(x)
        p = np.asarray(p, dtype=float) if len(p) > 0 else np.ones(x.shape) / x.shape[0]
        
        assert not np.isnan(p).any(), "p contains NaN values"

        # Ensure both x and p are 1D
        if x.ndim != 1 or p.ndim != 1:
            raise ValueError("Both x and p must be 1D lists or 1D numpy arrays.")

        # Ensure all elements in `x` are of the same type
        if all(isinstance(i, (int, float)) for i in x.tolist()) or all(isinstance(i, str) for i in x):
            pass
        else:
            raise TypeError("All elements in x must be either all numbers or all strings.")
        
        # Assert sum of values equal 1.
        if not np.isclose(sum(p), 1):
            raise ValueError("Sum of probabilities in diction must equal 1.")
        
        self.x = x
        self.p = p
        self.rg_type = 'choice'
        return self   # For chaining

    def discrete2continuous_pdf(self, x, p, new_del_x:float, new_x_min:float=None, new_x_max:float=None,  incr_x_init:float=0):
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
        p = np.asarray(p, dtype=float) if len(p) > 0 else np.ones_like(x) / len(x)
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
            
        self.x = new_x_array
        self.p = new_p
        self.rg_type = 'choice'
        return self   # For chaining
        
class RandomGenerator(RandomGeneratorDefine):
    def __init__(self, random_seed_generator=np.random.default_rng()):
        super().__init__(random_seed_generator)

    def generate(self, size: tuple = (1,)) -> float:
        """
        Generates a random number/string based on relevant RandomGeneratorDefine class/rg_type.

        Parameters:
        size (tuple): Shape of the list (or 2D array) to generate, e.g., (n,) for 1D list or (m, n) for 2D array.
        
        Returns:
        float: A random number/string if size is (1,)
        numpy array: if size is not so.
        """
        assert self.rg_type != "NA", "Random Generator has not been defined. Use either normal or uniform or discrete."

        if self.rg_type == 'uniform':
            rnd_gen = self.random_seed_generator.uniform(self.low, self.high, size=size)
        elif self.rg_type == 'normal':
            rnd_gen = self.random_seed_generator.normal(self.mean, self.std, size=size)
        elif self.rg_type == 'choice': #Discrete or discrete to continuous
            x = self.x
            p = self.p
            rnd_gen = self.random_seed_generator.choice(x, size=size, p=p)
        elif self.rg_type == 'constant':
            rnd_gen = np.ones(size) * self.x
        else:
            raise ValueError("Random Generator type is unrecognized. Use either normal or uniform or discrete.")

        # if size == (1,):
            # rnd_gen = rnd_gen[0]
        return rnd_gen
    
    def plot_pmf(self, ax=None, n=1000000, bins=100):
        
        if ax is None:
            fig, ax = plt.subplots()
        
        rnd_gen = self.generate(size=(n,))
        if self.rg_type != 'choice':
            ax.hist(rnd_gen, bins=bins, density=True)#, align='left')
            ax.set(xlabel='x',
                   ylabel='f(x)',
                   title='Probability Density Function (PDF)')
            
            if self.rg_type == 'uniform':
                ax.set_ylim([ax.get_ylim()[0], ax.get_ylim()[1]*5])
        else:
            x = self.x
            x = np.array([str(a) for a in x])
            rnd_gen = np.array([str(a) for a in rnd_gen])
            counts = [np.sum(rnd_gen == category)/len(rnd_gen) for category in x]
            ax.bar(x, counts)
            ax.set(xlabel='x',
                   ylabel='P(x)',
                   title='Probability Mass Function (PMF)',
                   ylim=[0,1])
        # plt.ylim([0,1])
        # plt.title('Binomial Distribution PMF (n=10, p=0.5)')
        return ax