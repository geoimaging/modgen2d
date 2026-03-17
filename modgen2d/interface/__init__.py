# This file is part of modgen2d, a Python package for 2D subsurface model generation.
# Copyright (C) XXXX Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
# LICENSE

"""
Public API for the 'modgen2d.interface' generated model subpackage.
"""

# PUBLIC API — this is the only file users ever see

from . import rough_interface_generator, interface_smoother, depth_updaters
from ._read_only import DiscretizedInterfaces2DReadOnly
from ._main import DiscretizedInterfaces2D
from ._from_dict import DiscretizedInterfaces2DFromDict
from .global_soil_interface_config import GlobalSoilInterfaceConfig

__all__ = [
    "rough_interface_generator", "interface_smoother", "depth_updaters",
    "DiscretizedInterfaces2D", "DiscretizedInterfaces2DReadOnly", "DiscretizedInterfaces2DFromDict", 
    "GlobalSoilInterfaceConfig",
]