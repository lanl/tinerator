![logo](docs/_static/logo-color-horiz.svg)

[![build](https://github.com/lanl/tinerator/actions/workflows/docker-image.yml/badge.svg)](https://github.com/lanl/tinerator/actions/workflows/docker-image.yml) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lanl/tinerator/master)

[![dockerhub](https://img.shields.io/static/v1?label=Docker&message=Download%20container&color=blue&style=for-the-badge&logo=docker)](https://hub.docker.com/r/ees16/tinerator) <br/>
[![readthedocs](https://img.shields.io/static/v1?label=Documentation&message=Read%20online&color=blue&style=for-the-badge&logo=read-the-docs)](https://lanl.github.io/tinerator/) <br/>
[![jupyter](https://img.shields.io/static/v1?label=Jupyter%20Notebook&message=View%20examples&color=blue&style=for-the-badge&logo=jupyter)](https://github.com/lanl/tinerator/tree/master/examples)

## About TINerator

**TINerator** is a Python module for the easy creation of unstructured 3D and 2.5D meshes from GIS data sources. Digital Elevation Maps (DEMs) can be quickly turned into unstructured triangulated meshes, and then further refined by the import of flowline shapefiles or automatically through watershed delineation. Advanced layering and attribute functions then allow for a complex subsurface to be defined. 

TINerator comes with a host of 2D and 3D visualization functions, allowing the user to view the status of the mesh at every step in the workflow. In addition, there are geometrical tools for removing triangles outside of a polygon, generating quality analytics on the mesh, adding cell- and node-based attributes to a mesh, and much more.

It was created at Los Alamos National Laboratory and funded primarily through the [IDEAS-Watersheds](https://ideas-productivity.org/ideas-watersheds/) and [NGEE Arctic](https://ngee-arctic.ornl.gov) programs. It has since been used by researchers and US-DOE national laboratories nation-wide.

TINerator has been designed to work well with [Amanzi-ATS](https://amanzi.github.io).

## Documentation

Documentation is available at [https://lanl.github.io/tinerator/](https://lanl.github.io/tinerator).

## Installing TINerator
#### Conda

TINerator is available on [Conda Forge](https://anaconda.org/conda-forge/tinerator):

```sh
$ conda install -c conda-forge tinerator
```

#### PyPI / `pip`

To install through [PyPI](https://pypi.org/project/tinerator/):

```sh
$ pip install tinerator
```

#### Source Code

```sh
$ git clone git@github.com:lanl/tinerator.git
$ cd tinerator/
$ python -m pip install ".[all]"
```

#### Docker

TINerator is available as a Docker container,
complete with Jupyter Lab integration, functioning
as a complete mesh generation environment:

```sh
$ docker pull ees16/tinerator:latest
```

## Troubleshooting
### Building GDAL

The most common point of failure is installing GDAL.
The Conda Forge recipe is the easiest way to get GDAL
working:

```sh
$ conda install gdal
```

If you have GDAL installed independently (i.e., through
`brew` or `apt`), you may install Python bindings through
PyPI:

```sh
$ pip install GDAL==`gdal-config --version`
```

## Contributing

Pull requests of all manner are welcomed! Please read the [contributing guidelines](CONTRIBUTING.md) before submitting a pull request.

