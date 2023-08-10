# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../../src/'))
sys.path.insert(0, os.path.abspath('../../src/pyfem'))
from pyfem import __version__

project = 'pyfem'
copyright = '2023, Sun Jingyu'
author = 'Sun Jingyu'
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    # 'sphinx.ext.mathjax',
    # 'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.duration',
    'sphinx-prompt',
    'sphinx_tabs.tabs',
    'sphinx.ext.extlinks',
    "sphinx_copybutton",
    "sphinx_design",
    # 'sphinxcontrib.httpdomain',
    # 'sphinxcontrib.video',
    'sphinxemoji.sphinxemoji',
    # 'sphinxext.opengraph',
    # 'm2r2'
]

intersphinx_mapping = {
    "rtd": ("https://docs.readthedocs.io/en/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}

intersphinx_disabled_domains = ["std"]

templates_path = ['_templates']

exclude_patterns = []

language = 'zh_CN'

math_number_all = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
# html_theme = 'classic'
# html_theme = 'furo'

# html_show_sphinx = False

html_static_path = ['_static']

html_style = 'css/custom.css'

source_suffix = ['.rst', '.md']
