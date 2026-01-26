# This file is part of geomodgen2d, a Python package for 2D subsurface model generation.
# Copyright (C) XXXX Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
# LICENSE

"""
This module exposes the public API for working with 2D lithological domains
in the geomodgen2d package. Users typically interact with these classes
to define, read, or manipulate lithological structures and collections
for subsurface modeling.

Public API
----------
- LithologicalDomain2D: Base class for 2D lithological domains.
- LithologicalDomain2DReadOnly: Read-only variant of a lithological domain.
- LithologicalDomain2DFromObstruction2D: Specialized class to generate
  domains based on 2D obstructions.
- LithologicalDomain2DCollection: Collection class managing multiple
  lithological domains with merging, ordering, and sampling support.

Notes
-----
This file is intended for user-facing imports only. Internal modules
(a_*, b_*) should not be imported directly by users.
"""

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