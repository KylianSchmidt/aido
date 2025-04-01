import os
import sys

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'aido'
copyright = '2024, Kylian Schmidt, Dr. Jan Kieseler'
author = 'Kylian Schmidt, Dr. Jan Kieseler'
release = "0.0.3"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

master_doc = "home"
extensions = [
    'sphinx.ext.autodoc',           # For automatical documentation
    'sphinx_autodoc_typehints',     # Autodoc typehints
    'sphinx.ext.napoleon',          # For Google and NumPy-style docstrings
    'sphinx.ext.mathjax',           # For LaTeX math rendering,
    'myst_parser',                  # For .md files
    "sphinx.ext.intersphinx",       # For genindex and other files
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

# Add markdown files with myst_parser
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown"
}
sys.path.insert(0, os.path.abspath("../aido"))
myst_enable_extensions = ["deflist", "colon_fence"]

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": True,
    "show-inheritance": True
}
