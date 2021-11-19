![logo](docs/_static/logo-color-horiz.svg)

[![build](https://github.com/lanl/tinerator/actions/workflows/docker-image.yml/badge.svg)](https://github.com/lanl/tinerator/actions/workflows/docker-image.yml) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lanl/tinerator/master)

[![dockerhub](https://img.shields.io/static/v1?label=Docker&message=Download%20container&color=blue&style=for-the-badge&logo=docker)](https://hub.docker.com/r/ees16/tinerator) <br/>
[![readthedocs](https://img.shields.io/static/v1?label=Documentation&message=Read%20online&color=blue&style=for-the-badge&logo=read-the-docs)](https://lanl.github.io/tinerator/) <br/>
[![jupyter](https://img.shields.io/static/v1?label=Jupyter%20Notebook&message=View%20examples&color=blue&style=for-the-badge&logo=jupyter)](https://github.com/lanl/tinerator/tree/master/examples)

### About TINerator

**TINerator** is a Python module for the easy creation of unstructured 3D and 2.5D meshes from GIS data sources. Digital Elevation Maps (DEMs) can be quickly turned into unstructured triangulated meshes, and then further refined by the import of flowline shapefiles or automatically through watershed delineation. Advanced layering and attribute functions then allow for a complex subsurface to be defined. 

TINerator comes with a host of 2D and 3D visualization functions, allowing the user to view the status of the mesh at every step in the workflow. In addition, there are geometrical tools for removing triangles outside of a polygon, generating quality analytics on the mesh, adding cell- and node-based attributes to a mesh, and much more.

It was created at Los Alamos National Laboratory and funded primarily through the [IDEAS-Watersheds](https://ideas-productivity.org/ideas-watersheds/) and [NGEE Arctic](https://ngee-arctic.ornl.gov) programs. It has since been used by researchers and US-DOE national laboratories nation-wide.

TINerator has been designed to work well with [Amanzi-ATS](https://amanzi.github.io).

### Quick Start

TINerator is available on [Conda Forge](https://anaconda.org/conda-forge/tinerator):

```sh
$ conda install -c conda-forge tinerator
```

as well as [PyPI](https://pypi.org/project/tinerator/):

```sh
$ pip install tinerator
```

Alternatively, you can build directly from this repo:

```sh
$ git clone git@github.com:lanl/tinerator.git
$ cd tinerator/
$ python -m pip install .[test] -vv
```

#### INSTALLATION NOTE:

The most common point-of-failure installing TINerator is 
installation/build errors with GDAL.

Before installing TINerator, it is recommended that you install GDAL first.

##### (RECOMMENDED) Building GDAL through Conda

```sh
$ conda install gdal
```

##### Installing GDAL through `pip`

This method requires (i) building the GDAL system libraries, and (ii) installing the Python wrapping module.

```sh
# Installing GDAL libs
$ brew install gdal # on macOS
$ apt-get install -y gdal-bin libgdal-dev # Ubuntu/Debian Linux

# Installing Python module - 
#   explicitly install wrappers for the version
#   of GDAL built above
$ pip install GDAL==`gdal-config --version`
```

If the above fails, you may have success with PyGDAL:

```sh
$ pip install pygdal
```

### Documentation

- [Read the documentation online](https://lanl.github.io/tinerator)

### Quick Start

#### Online Demo

[![Binder](https://mybinder.org/badge_logo.svg)]()

You can run TINerator Jupyter notebooks online with [MyBinder](https://mybinder.org/v2/gh/lanl/tinerator/master). It may take a few minutes for the container to initialize.

#### Docker Container

The easiest way to get started with TINerator is through [Docker](https://hub.docker.com/r/ees16/tinerator):

    $ docker pull ees16/tinerator:latest
    $ docker run -it \
        -v $(pwd):/docker_user/work \
        -p 8899:8899 \
        -p 8050:8050 \
        ees16/tinerator:latest

After the container launches, navigate to `http://127.0.0.1:8899/lab` in a web browser to begin using TINerator within a Jupyter Lab instance. Example notebooks and HTML documentation are available within Jupyter.

### Building TINerator

To build TINerator from source, refer to the [documentation](https://lanl.github.io/tinerator/installation.html).

#### Linux

```sh
$ apt-get install --no-install-recommends -y \
  g++ \
  gfortran \
  make \
  cmake \
  libgl1-mesa-glx \
  \
  # NOTE: gdal on conda forge is *strongly* preferred
  # to the apt version.
  libdal-dev
```

#### macOS

```sh
$ brew install \
    gdal \
    gcc \
    gfortran
```

Note that the module has not been tested on Apple Silicon yet.

### Contributing

Pull requests of all manner are welcomed! Please read the [contributing guidelines](CONTRIBUTING.md) before submitting a pull request.

