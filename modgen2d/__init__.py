# This file is part of modgen2d, a Python package for 2D subsurface model generation.
# Copyright (C) XXXX Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
# LICENSE

"""Import modules into the modgen2d namespace."""

from .metadata import __version__

# Core modules
from .units_config import Units
from .discretized_domain2d import DiscretizedDomain2D
from .discretized_interfaces2d import DiscretizedInterfaces2D
from .discretized_interfaces2d_from_dict import DiscretizedInterfaces2DFromDict 
from .features_config import FeaturesConfig
from .global_soil_interface_config import GlobalSoilInterfaceConfig
from .generated_model2d import GeneratedModel2D, GeneratedModel2DMerged, GeneratedProfileCollection2DReadOnly, GeneratedProfileCollection2D
from .lithological_domain2d import LithologicalDomain2D, LithologicalDomain2DCollection, LithologicalDomain2DFromObstruction2D, LithologicalDomain2DReadOnly
from .main_properties import MainPropertiesConfig, AuxillaryProperties
from .property_distribution import PropertyDistribution
from .main_property_each import MainProperty
from .obstruction2d import Obstruction2D
from .random_generators import RandomGeneratorAbstract, Constant, LogUniform, Uniform, Normal, DiscreteChoice
from .rough_interface_creator2d import AbstractRoughInterfaceCreator, NormalInterfaceGen, UniformInterfaceGen
from .spatial_simulator2d import SpatialSimulator2D, ConstantSimulator, CovarianceDecompositionSimulator
from .load_generated_model2d_from_hdf5 import load_dict_from_hdf5, read_hdf5_file

# Expose a flat namespace for docs and imports
__all__ = [
    # Core
    "Units",
    "DiscretizedDomain2D",
    "DiscretizedInterfaces2D",
    "DiscretizedInterfaces2DFromDict",
    "FeaturesConfig",
    "GlobalSoilInterfaceConfig",
    "GeneratedModel2D",
    "GeneratedModel2DMerged",
    "GeneratedProfileCollection2DReadOnly",
    "GeneratedProfileCollection2D",
    "LithologicalDomain2D",
    "LithologicalDomain2DCollection",
    "LithologicalDomain2DFromObstruction2D",
    "LithologicalDomain2DReadOnly",
    "MainPropertiesConfig",
    "AuxillaryProperties",
    "MainProperty",
    "PropertyDistribution",
    "Obstruction2D",
    "RandomGeneratorAbstract",
    "Constant",
    "LogUniform",
    "Uniform",
    "Normal",
    "DiscreteChoice",
    "AbstractRoughInterfaceCreator",
    "NormalInterfaceGen",
    "UniformInterfaceGen",
    "SpatialSimulator2D",
    "ConstantSimulator",
    "CovarianceDecompositionSimulator",
    "load_dict_from_hdf5",
    "read_hdf5_file",
]