# This file is part of geomodgen2d, a Python package for 2D subsurface model generation.
# Copyright (C) XXXX Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
# LICENSE

"""Import modules into the geomodgen2d namespace."""

from .metadata import __version__

# Core modules
from . import discretized_domain2d
from . import discretized_interfaces2d
from . import features_config
from . import generated_model2d_collection
from . import lithological_domain2d
from . import lithological_domain2d_collection
from . import main_properties
from . import main_property_each
from . import obstruction2d
from . import random_generators
from . import rough_interface_creator2d
from . import spatial_simulator2d
from . import units_config

# Optional: control what gets imported when using 'from geomodgen2d import *'
__all__ = [
    "discretized_domain2d",
    "discretized_interfaces2d",
    "features_config",
    "generated_model2d_collection",
    "lithological_domain2d",
    "lithological_domain2d_collection",
    "main_properties",
    "main_property_each",
    "obstruction2d",
    "packages_ModelGenerator",
    "random_generators",
    "rough_interface_creator2d",
    "spatial_simulator2d",
    "units_config",
]
