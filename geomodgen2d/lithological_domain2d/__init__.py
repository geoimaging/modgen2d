# This file is part of geomodgen2d, a Python package for 2D subsurface model generation.
# Copyright (C) XXXX Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
# LICENSE

"""Import modules into the geomodgen2d namespace."""

# PUBLIC API — this is the only file users ever see

from .a_from_interface import LithologicalDomain2D
from .a_base import LithologicalDomain2DReadOnly
from .a_from_obs2d import LithologicalDomain2DFromObstruction2D
from .b_collection import LithologicalDomain2DCollection

__all__ = [
    "LithologicalDomain2D",
    "LithologicalDomain2DReadOnly",
    "LithologicalDomain2DFromObstruction2D",
    "LithologicalDomain2DCollection",
]