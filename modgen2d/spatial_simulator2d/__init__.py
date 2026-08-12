# This file is part of modgen2d, a Python package for 2D subsurface model generation.
# Copyright (C) XXXX Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
# LICENSE

"""
Public API for the 'modgen2d.spatial_simulator2d' generated model subpackage.

Only the simulator classes imported and exported here are part of the stable
public API.

Classes
-------
SpatialSimulator2DAbstract
    Abstract base class shared by all spatial simulators.
ConstantSimulator
    Deterministic simulator that evaluates only the prescribed mean trend.
CovarianceDecompositionSimulator
    Gaussian random-field simulator based on covariance decomposition.
GSToolsSimulator
    Scalable Gaussian random-field simulator based on GSTools.
load_simulator_from_config
    Function that load the correct spatial simulator.
"""

# PUBLIC API — this is the only file users ever see

from .a_abstract import SpatialSimulator2DAbstract
from .b_constant import ConstantSimulator
from .b_covdec import CovarianceDecompositionSimulator
from .b_gstools import GSToolsSimulator
from .c_loader import load_simulator_from_config

__all__ = [
    "SpatialSimulator2DAbstract",
    "ConstantSimulator",
    "CovarianceDecompositionSimulator",
    "GSToolsSimulator",
    "load_simulator_from_config",
]