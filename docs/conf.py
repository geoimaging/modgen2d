import os
import sys

sys.path.insert(0, os.path.abspath(".."))


project = 'geomodgen2d'
author = 'Sanish Bhochhibhoya and Joseph P. Vantassel'
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autodoc_mock_imports = [
    "geomodgen2d.global_soil_interface_config"
]

napoleon_google_docstring = False
napoleon_numpy_docstring = True

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_class_signature = "mixed"

html_theme = "sphinx_rtd_theme"
