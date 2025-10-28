import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'aido'
copyright = '2025, Kylian Schmidt, Jan Kieseler'
author = 'Kylian Schmidt, Jan Kieseler'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

master_doc = "index"
extensions = [
    'autoapi.extension',            # For automatical documentation
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

html_theme = "furo"
html_static_path = ['_static']
html_css_files =  ["html_style.css"]
html_theme_options = {
    "navigation_with_keys": True,
    "sidebar_hide_name": False,
}

# Add markdown files with myst_parser
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown"
}
sys.path.insert(0, os.path.abspath(".."))
myst_enable_extensions = ["deflist", "colon_fence"]

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
    'docutils.block_quote_ends_without_blank_line',  # Suppress block quote warnings
    'autoapi.python_import_resolution',  # Suppress autoapi duplicate warnings
    "autoapi.toc_reference",
]

# Napoleon settings for better docstring parsing
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autoapi settings
autoapi_type = "python"
autoapi_dirs = ["../aido"]
autoapi_options = [
    "members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
autoapi_member_order = "bysource"
autoapi_python_class_content = "class"
autoapi_root = "api"
autoapi_keep_files = True
autoapi_add_toctree_entry = True
autoapi_python_use_implicit_namespaces = True
autoapi_add_objects_to_toctree = True
