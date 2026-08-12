"""Concrete spatial simulators for 2D lithological domains."""

import numpy as np
from .a_abstract import SpatialSimulator2DAbstract

class CovarianceDecompositionSimulator(SpatialSimulator2DAbstract):
    """
    Spatial simulator using covariance decomposition.

    Generates Gaussian random fields using exponential correlation
    functions and Cholesky decomposition.
    
    Parameters
    ----------
    theta_x : float or None
        Correlation length in the x-direction.
    theta_z : float or None
        Correlation length in the z-direction.
    simulated_val_for_ignored_lit_property : int, default=-99999
        Constant value assigned to ignored lithological IDs.
    rng : numpy.random.Generator, optional
        Random number generator.
    """
    def __init__(self, theta_x, theta_z, simulated_val_for_ignored_lit_property=-99999, rng=np.random.default_rng()):
        
        # Validate theta_x and theta_z
        if not isinstance(theta_x, (int, float)):
            raise TypeError("theta_x must be float.")
        if not isinstance(theta_z, (int, float)):
            raise TypeError("theta_z must be float.")
            
        params = {'theta_x': theta_x,
                  'theta_z': theta_z,
                 }
        
        super().__init__(params, simulated_val_for_ignored_lit_property, rng)
        self.reloadable=True  #To use in loading gen_model_collection from config, so dont use True for any other case.
    
    def _compute_correlation_matrix(self, points):
        """
        Compute the Cholesky factor of correlation matrix R. Note: C = sigma^2 * R,
        where R is the correlation matrix determined by theta_x/theta_z.
        Returns L such that C = L @ L.T
        """
        pts = self.check_points(points)
        
        x = pts[:, 0][:, None]  # (N,1)
        z = pts[:, 1][:, None]  # (N,1)

        # Pairwise separations
        dx = np.abs(x - x.T)
        dz = np.abs(z - z.T)

        theta_x = self.params['theta_x']
        theta_z = self.params['theta_z']

        # Correlation matrix (σ=1)
        R = np.exp(-2 * dx / theta_x) * np.exp(-2 * dz / theta_z)

        # Cholesky of correlation matrix
        L_R = np.linalg.cholesky(R)
        
        return L_R

    def simulate(self, points, mean=0, sigma=1):
        """
        mean : float or sequence of 2 floats, default=0
            mean = a_m + b_m*z
            - If scalar → use this mean value for the entire field. b_m = 0
            - If sequence of 2 numbers → [a_m, b_m]
            
        sigma : float
            Standard deviation scaling for the field.
        """
        # Step 1: Compute linear mean trend
        pts = self.check_points(points)
        
        a_m, b_m = self.get_means_am_bm(mean)
        z = pts[:, 1]
        mean_vector = a_m + b_m * z
        
        if sigma==0:
            return mean_vector
        
        # Step 2: Compute Cholesky of correlation/covariance
        L = self._compute_correlation_matrix(pts)  # shape (N,N)

        # Step 3: Generate standard normal vector
        u = self.rng.standard_normal(pts.shape[0])   # shape (N,)

        # Step 4: Multiply to get correlated deviations
        correlated_field = L @ u                        # shape (N,)

        # Step 5: Scale by sigma and add mean trend
        #simulated_vector =  Mean + correlated_field * sigma
        simulated_field = mean_vector + correlated_field * sigma
        return simulated_field