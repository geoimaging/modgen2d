# This file is part of geomodgen3d a Python package for ...
# Copyright (C) XXXX Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
# LICENSE

"""Define a three-dimensional domain that defines a material."""

import numpy as np

from geomodgen3d.domain3d import Domain3D

class MaterialDomain3D(Domain3D):

    # TODO(jpv): Need to develop a convention for cardinal directions
    def __init__(self, material, dx, dy, dz):
        span = material.shape
        super().__init__(*span, dx, dy, dz)
        self.material = np.array(material).astype(float)
