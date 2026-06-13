modgen2d
========

modgen2d is a Python package for generating 2D subsurface models with customizable soil properties, utilities, interfaces, lithological domains, and spatial correlation.

Core modules
------------

.. toctree::
   :maxdepth: 2

   api/length_config
   api/discretized_domain2d
   api/features_config
   api/obstruction_2d
   api/random_generators
   api/main_property
   api/auxillary_properties
   api/main_properties_config
   api/spatial_simulator2d


Interface modules
-----------------

.. toctree::
   :maxdepth: 2

   api/interface_main
   api/interface_from_dict
   api/interface_rough_interface_generator
   api/interface_smoother
   api/interface_depth_updaters
   api/interface_global_soil_interface_config


Lithological domain modules
---------------------------

.. toctree::
   :maxdepth: 2

   api/lithological_domain2d
   api/lithological_domain2d_from_obs
   api/lithological_domain2d_collection

Generated model modules
-----------------------

.. toctree::
   :maxdepth: 2

   api/generated_model2d
   api/generated_model2d_merged
   api/generated_model2d_collections_readonly
   api/generated_model2d_collections

Load Generated Model2D
----------------------

.. toctree::
   :maxdepth: 2

   api/load_generated_model2d_from_hdf5