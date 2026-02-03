import modgen2d as mg
import numpy as np
from scipy.stats import multivariate_normal, norm

class CorrelatedUniformBivariateXLogY(mg.RandomGeneratorAbstract):
    def __init__(self, u_low_x, u_high_x, u_low_logy, u_high_logy, r_x_logy, rng=None):
        """
        Initializes a correlated uniform random generator using a Gaussian copula.

        X  ~ Uniform(u_low_x, u_high_x)
        log10(Y) ~ Uniform(log10(u_low_logy), log10(u_high_logy))
        Corr(X, log10(Y)) = r_x_logy
        
        Parameters
            u_low_x, u_high_x : float :  Range for Variable1.
            u_low_logy, u_high_logy : float : Range for Variable2.
            r : float : Correlation coefficient between Variable 1 and Variable 2.
            r_comp_axis: string : Axes scale for correlation computation.
                First element must be in {"x", "log10x"}, second in {"y", "log10y"}.
                That is Options: ['x', 'y'], ['log10x', 'log10y'], ['log10x', 'y'], ['x', 'log10y']]
        """
        self.u_low_x = u_low_x
        self.u_high_x = u_high_x
        self.u_low_logy = u_low_logy
        self.u_high_logy = u_high_logy
        
        # --- means checks ---
        if u_low_x > u_high_x:
            raise ValueError("u_low_x must be < u_high_x")
        if u_low_logy <= 0 or u_high_logy <= 0:
            raise ValueError("Log-uniform variable must be strictly positive.")
        if u_low_logy > u_high_logy:
            raise ValueError("u_low_logy must be < u_high_logy")

        # --- correlation checks ---
        if not isinstance(r_x_logy, (int, float)):
            raise TypeError("Correlation coefficient r must be a float or int.")
        if not -1 <= r_x_logy <= 1:
            raise ValueError(f"Correlation coefficient r must be between -1 and 1, got {r_x_logy}")
            
        self.r_x_logy = r_x_logy

    def generate(self, size=1):
        if isinstance(size, int):
            n = size
            out_shape = (size, 2)
        else:
            raise TypeError("size must be int")
            
        cov = [[1, self.r_x_logy], [self.r_x_logy, 1]]
        z = multivariate_normal.rvs(mean=[0, 0], cov=cov, size=size)
        u = norm.cdf(z)
        u = u.reshape(-1, 2)
        
        x = self.u_low_x + (self.u_high_x - self.u_low_x) * u[:, 0]
        logy = np.log10(self.u_low_logy) + (np.log10(self.u_high_logy) - np.log10(self.u_low_logy)) * u[:, 1]
        y = 10 ** logy
       
        samples = np.column_stack((x, y))
        return samples.reshape(out_shape)
