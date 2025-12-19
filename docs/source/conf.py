# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

os.environ["QT_QPA_PLATFORM"] = "offscreen"

sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "mne-videobrowser"
copyright = "2025, Aalto University"  # noqa: A001
author = "Teemu Taivainen"
release = "0.1.3"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
]

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}

# Napoleon settings for Google and NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# Autosummary settings
autosummary_generate = True

# Intersphinx mapping to link to other projects
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "mne": ("https://mne.tools/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

# Theme options
html_theme_options = {
    "logo": {
        "text": "mne-videobrowser",
    },
    "github_url": "https://github.com/ttaiv/mne-videobrowser",
    "navigation_depth": 4,
    "show_nav_level": 0,
    "show_toc_level": 2,
}


def setup(app):
    """Configure Sphinx extension hooks.

    This function is called by Sphinx during the build process. We use it here
    to register a custom handler for the 'autodoc-skip-member' event. This
    handler prevents the automatic documentation of Qt Signal objects, which
    have docstrings that cause Sphinx warnings (due to unclosed emphasis).
    """
    try:
        from qtpy.QtCore import Signal  # pyright: ignore[reportPrivateImportUsage]
    except ImportError:
        return

    def skip_signal(app, what, name, obj, skip, options):
        if isinstance(obj, Signal):
            return True
        return skip

    app.connect("autodoc-skip-member", skip_signal)
