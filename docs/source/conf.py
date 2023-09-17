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
    'sphinx.ext.mathjax',
    # 'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.duration',
    'sphinx.ext.extlinks',
    'sphinx-prompt',
    'sphinx_tabs.tabs',
    'sphinx_copybutton',
    'sphinx_design',
    'hoverxref.extension',
    # 'sphinxcontrib.httpdomain',
    # 'sphinxcontrib.video',
    'sphinxemoji.sphinxemoji',
    # 'sphinxext.opengraph',
    # 'm2r2'
]

# autodoc_default_options = {
#     'members': True,
#     'undoc-members': True,
#     'private-members': True,
#     'special-members': True,
#     'inherited-members': True,
#     'show-inheritance': True,
#     'member-order': 'bysource',
# }

# intersphinx_mapping = {
#     "rtd": ("https://docs.readthedocs.io/en/stable/", None),
#     "python": ("https://docs.python.org/3/", None),
#     "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
# }

templates_path = ['_templates']

exclude_patterns = []

language = 'zh_CN'

# -- Options for math --------------------------------------------------------

numfig = True  # 设置公式编号跨页

math_number_all = True  # 启用公式编号

math_numfig = True  # 设置公式编号跨页

math_eqref_format = '({number})'

numfig_secnum_depth = 3

mathjax_path = 'https://cdn.staticfile.org/mathjax/3.2.2/es5/tex-mml-chtml.js'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
# html_theme = 'classic'
# html_theme = 'furo'

html_show_sphinx = False

html_static_path = ['_static']

html_style = 'css/custom.css'

source_suffix = ['.rst', '.md']

# html_logo = '_static/img/logo.ico'

# -- Options for hoverxref ---------------------------------------------------

hoverxref_intersphinx = [
    "sphinx",
    "pip",
    "nbsphinx",
    "myst-nb",
    "ipywidgets",
    "jupytext",
]

hoverxref_auto_ref = True
hoverxref_domains = ["py"]
hoverxref_roles = [
    'mod',
    'class',
    'ref',
    'numref',
    "option",
    "doc",
    "term",
]

hoverxref_role_types = {
    "mod": "modal",  # for Python Sphinx Domain
    "doc": "modal",  # for whole docs
    "class": "tooltip",  # for Python Sphinx Domain
    "ref": "tooltip",  # for hoverxref_auto_ref config
    "confval": "tooltip",  # for custom object
    "term": "tooltip",  # for glossaries
    'numref': 'tooltip',
}
