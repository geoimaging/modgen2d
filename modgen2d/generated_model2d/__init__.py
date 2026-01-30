# This file is part of modgen2d, a Python package for 2D subsurface model generation.
# Copyright (C) XXXX Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
# LICENSE

"""
Public API for the 'modgen2d' generated model subpackage.

This module defines the user-facing interface for accessing generated
2D subsurface models and collections. Only the classes imported and
exported here are considered part of the stable public API.

Classes
-------
GeneratedModel2D
    Represents a single generated 2D subsurface model.
GeneratedProfileCollection2DReadOnly
    Read-only collection of generated 2D profiles.
GeneratedProfileCollection2D
    Mutable collection of generated 2D profiles.
GeneratedModel2DMerged
    Merged representation of multiple generated 2D models.
"""

# PUBLIC API — this is the only file users ever see

from .a_each import GeneratedModel2D
from .b_collection import GeneratedProfileCollection2DReadOnly, GeneratedProfileCollection2D
from .c_merged import GeneratedModel2DMerged

__all__ = [
    "GeneratedModel2D",
    "GeneratedProfileCollection2DReadOnly",
    "GeneratedProfileCollection2D",
    "GeneratedModel2DMerged",
]