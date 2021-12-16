# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "TINerator"
copyright = "2021, Daniel Livingston"
author = "Daniel Livingston"

# The full version, including alpha/beta/rc tags
release = "1.0.0"

# -- General configuration ---------------------------------------------------

needs_sphinx = "1.0"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.coverage",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_inline_tabs",
    "sphinx_gallery.gen_gallery",
    "nbsphinx"
]

sphinx_gallery_conf = {
    "examples_dirs": "../examples",  # path to your example scripts
    "gallery_dirs": "examples-py",  # path to where to save gallery generated output
}

# This is processed by Jinja2 and inserted before each notebook
nbsphinx_prolog = r"""
{% set docname = 'doc/' + env.doc2path(env.docname, base=None) %}
.. raw:: html
    <div class="admonition note">
      This page was generated from
      <a class="reference external" href="https://github.com/spatialaudio/nbsphinx/blob/{{ env.config.release|e }}/{{ docname|e }}">{{ docname|e }}</a>.
      Interactive online version:
      <span style="white-space: nowrap;"><a href="https://mybinder.org/v2/gh/spatialaudio/nbsphinx/{{ env.config.release|e }}?filepath={{ docname|e }}"><img alt="Binder badge" src="https://mybinder.org/badge_logo.svg" style="vertical-align:text-bottom"></a>.</span>
      <script>
        if (document.location.host) {
          $(document.currentScript).replaceWith(
            '<a class="reference external" ' +
            'href="https://nbviewer.jupyter.org/url' +
            (window.location.protocol == 'https:' ? 's/' : '/') +
            window.location.host +
            window.location.pathname.slice(0, -4) +
            'ipynb">View in <em>nbviewer</em></a>.'
          );
        }
      </script>
    </div>
.. raw:: latex
    \nbsphinxstartnotebook{\scriptsize\noindent\strut
    \textcolor{gray}{The following section was generated from
    \sphinxcode{\sphinxupquote{\strut {{ docname | escape_latex }}}} \dotfill}}
"""

# This is processed by Jinja2 and inserted after each notebook
nbsphinx_epilog = r"""
{% set docname = 'doc/' + env.doc2path(env.docname, base=None) %}
.. raw:: latex
    \nbsphinxstopnotebook{\scriptsize\noindent\strut
    \textcolor{gray}{\dotfill\ \sphinxcode{\sphinxupquote{\strut
    {{ docname | escape_latex }}}} ends here.}}
"""

# Don't add .txt suffix to source files:
#html_sourcelink_suffix = ''

# List of arguments to be passed to the kernel that executes the notebooks:
#nbsphinx_execute_arguments = [
#    "--InlineBackend.figure_formats={'svg', 'pdf'}",
#    "--InlineBackend.rc=figure.dpi=96",
#]

print("Copy example notebooks into docs/_examples")

def all_but_ipynb(dir, contents):
    result = []
    for c in contents:
        if os.path.isfile(os.path.join(dir,c)) and (not c.endswith(".ipynb")):
            result += [c]
    return result

def copy_notebooks(notebook_dir, target_dir):
    import shutil

    for root, dir, files in os.walk(notebook_dir):
        for file in files:
            if file.lower().endswith('.ipynb'):
                new_dir = root.replace(notebook_dir, target_dir)
                print('>>', root, notebook_dir, target_dir, new_dir)
                os.makedirs(new_dir, exist_ok=True)
                shutil.copyfile(os.path.join(root, file), os.path.join(new_dir, file))

script_path = os.path.realpath(__file__).replace('conf.py', '')
notebook_source_path = os.path.join(script_path, '..', 'examples/', 'notebooks/')
notebook_target_path = os.path.join(script_path, 'workflows/')
copy_notebooks(notebook_source_path, notebook_target_path)

source_suffix = ".rst" #[".rst", ".ipynb"]
master_doc = "index"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"  # https://pradyunsg.me/furo/

html_theme_options = {
    "light_logo": "logo-color-vert.svg",
    "dark_logo": "logo-color-vert.svg",
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
    #"announcement": "&#x1F389; <em>TINerator v1.0.0</em> has been released! &#x1F389;",
}


# pygments_style = "sphinx"
pygments_style = "monokai"
pygments_dark_style = "monokai"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_favicon = "_static/favicon.ico"
