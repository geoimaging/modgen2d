Introduction to `geomodgen2d`
=============================

About `geomodgen2d`
-------------------

`geomodgen2d` is an open-source Python package for generating
two-dimensional subsurface models for geotechnical and geophysical
applications. The package provides a flexible, data-driven framework
for simulating layered soil profiles, rough interfaces, utilities,
and spatially variable material properties.

`geomodgen2d` is developed by **Sanish Bhochhibhoya** and
**Joseph P. Vantassel**, and is designed to support research and
education in subsurface characterization, numerical simulation,
and uncertainty-aware modeling workflows.

The package is particularly well-suited for applications involving
synthetic data generation, seismic and electromagnetic wave
propagation studies, and AI-driven subsurface imaging.

Citation
--------

If you use `geomodgen2d` in your research or publications, please cite
the software appropriately. A recommended citation will be provided
here once an archival release is available.

.. note::
   For software, version-specific citations should be preferred when
   available. If using a development version of `geomodgen2d`, please
   cite the associated repository and commit hash.

Key Features
------------

`geomodgen2d` provides capabilities not commonly available in a single
model-generation framework, including:

- Programmatic generation of 2D subsurface domains with user-defined
  geometry and resolution
- Support for layered media, rough interfaces, and embedded utilities
- Spatially correlated and uncorrelated material property simulation
- Modular configuration of primary and auxiliary properties
- Seamless integration with downstream numerical simulators
- Reproducible model generation using configuration-driven workflows

Intended Use
------------

`geomodgen2d` is intended for:

- Synthetic model generation for seismic and electromagnetic simulations
- Sensitivity and uncertainty analyses
- Method development for inverse problems and imaging
- Teaching and demonstration of subsurface modeling concepts

The package is designed to be extensible, allowing users to incorporate
custom property models, random field generators, and domain definitions.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   licensing
   geomodgen2d

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
