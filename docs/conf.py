import os
import sys

sys.path.insert(0, os.path.abspath(".."))

from aido import __version__  # noqa

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'aido'
copyright = '2024, Kylian Schmidt, Dr. Jan Kieseler'
author = 'Kylian Schmidt, Dr. Jan Kieseler'
release = __version__

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
    "sphinx.ext.viewcode",          # Reference code
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
sys.path.insert(0, os.path.abspath(".."))
myst_enable_extensions = ["deflist", "colon_fence"]

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
}

# MyST settings
myst_enable_extensions = [
    "deflist",
    "colon_fence",
    "dollarmath",
    "amsmath"
]

# Suppress warnings
suppress_warnings = [
    'myst.xref_missing',
    'ref.python',
    'toc.excluded',  # Suppress toctree warnings
    'toc.not_readable',
    'toc.no_title',  # Suppress title warnings
    'docutils.definition_list_ends_without_blank_line',  # Suppress definition list warnings
    'docutils.bullet_list_ends_without_blank_line',  # Suppress bullet list warnings
    'docutils.block_quote_ends_without_blank_line'  # Suppress block quote warnings
]
