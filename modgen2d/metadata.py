# This file is part of geomodgen3d a Python package for ...
# Copyright (C) XXXX Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
# LICENSE

"""Metadata for modgen2d."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("modgen2d")
except PackageNotFoundError:
    __version__ = "unknown"
