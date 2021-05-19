![logo](docs/_static/logo-color-horiz.svg)

[![build](https://github.com/daniellivingston/tinerator-core/actions/workflows/docker-image.yml/badge.svg)](https://github.com/daniellivingston/tinerator-core/actions/workflows/docker-image.yml) [![docs](https://github.com/daniellivingston/tinerator-core/actions/workflows/gh-pages.yml/badge.svg)](https://github.com/daniellivingston/tinerator-core/actions/workflows/gh-pages.yml) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

### About TINerator

**TINerator** is a Python module for the easy creation of unstructured 3D and 2.5D meshes from GIS data sources. Digital Elevation Maps (DEMs) can be quickly turned into unstructured triangulated meshes, and then further refined by the import of flowline shapefiles or automatically through watershed delineation. Advanced layering and attribute functions then allow for a complex subsurface to be defined.

It was created at Los Alamos National Laboratory and funded primarily through the [IDEAS-Watersheds](https://ideas-productivity.org/ideas-watersheds/) and [NGEE Arctic](https://ngee-arctic.ornl.gov) programs. It has since been used by researchers and US-DOE national laboratories nation-wide.

TINerator has been designed to work well with [Amanzi-ATS](https://amanzi.github.io).

### Documentation

- [Read the documentation online](https://daniellivingston.github.io/tinerator-core)

### Quick Start

The easiest way to get started with TINerator is through [Docker](https://www.docker.com/):

    $ docker pull ees16/tinerator
    $ docker run -it ees16/tinerator

A Jupyter Lab instance will launch, and you will have the option to view example notebooks.

### Building TINerator

To build TINerator from source, refer to the [documentation](#).

### Contributing

Pull requests of all manner are welcomed! Please read the [contributing guidelines](#) before submitting a pull request.
