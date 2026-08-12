"""Concrete spatial simulators for 2D lithological domains."""

import numpy as np
from .a_abstract import SpatialSimulator2DAbstract

class ConstantSimulator(SpatialSimulator2DAbstract):
    """
    Deterministic non-spatial simulator.

    Produces values using only the specified mean function.

    Parameters
    ----------
    simulated_val_for_ignored_lit_property : int, default=-99999
        Constant value assigned to ignored lithological IDs.
    rng : numpy.random.Generator, optional
        Random number generator.
    """
    def __init__(self, simulated_val_for_ignored_lit_property=-99999, rng=np.random.default_rng()):
        super().__init__({}, simulated_val_for_ignored_lit_property, rng)
    
    def simulate(self, points, mean=0, sigma=None):
        a_m, b_m = self.get_means_am_bm(mean)    
        pts = self.check_points(points)
        z = pts[:, 1]      # extract z column
        simulated_f = a_m + b_m * z   # mean function: m(z) = a_m + b_m * z
        return simulated_f