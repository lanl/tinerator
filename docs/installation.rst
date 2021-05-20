Installation
============

Docker
------

The easiest way of using TINerator is through
`docker <https://www.docker.com/>`_, where you can 
``pull`` a `pre-built image of TINerator <https://hub.docker.com/r/ees16/tinerator/>`_ from
dockerhub::

    $ docker run -it --publish 8888:8888 --volume $(pwd):/tinerator/playground/work ees16/tinerator

When run, the Docker container initializes a Jupyter Lab instance.

You will see some output similar to this:

    [I 2021-05-20 22:48:24.778 LabApp] JupyterLab application directory is /usr/local/share/jupyter/lab
    [I 2021-05-20 22:48:24.782 ServerApp] jupyterlab | extension was successfully loaded.
    [I 2021-05-20 22:48:24.783 ServerApp] Serving notebooks from local directory: /tinerator/playground
    [I 2021-05-20 22:48:24.783 ServerApp] Jupyter Server 1.8.0 is running at:
    [I 2021-05-20 22:48:24.783 ServerApp] http://dae7d8d45c90:8888/lab
    [I 2021-05-20 22:48:24.783 ServerApp]     http://127.0.0.1:8888/lab
    [I 2021-05-20 22:48:24.784 ServerApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).

Open an internet browser on your machine and navigate to `http://127.0.0.1:8888/lab`. If all is successful, 
the Jupyter Lab logo should appear, along with a side panel showing `docs` and `examples` folders.

Clarifying the Docker command
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The above `docker run` command requires a lot of flags. They are:

- `-it`: short for `--interactive` + `--tty`
- `--publish 8888:8888`: Maps the network port 8888 within the Docker container
  to the network port 8888 on your computer. This is what allows you to 
  access the Jupyter Lab that is contained within Docker from your internet 
  browser.
- `--volume $(pwd):/tinerator/playground/work`: Docker cannot view files or folders on your machine
  unless it's explicitly allowed to via this command. This takes your current working directory
  `$(pwd)` and makes it visible from within the Docker container, at the location `/tinerator/playground/work`.

Download the Source
-------------------

The source code can be checked out from
`GitHub <https://github.com/daniellivingston/tinerator>`_::

    $ git clone https://github.com/daniellivingston/tinerator-core.git
    $ cd tinerator-core/

Dependencies
------------

Python module dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~

Third-party programs
~~~~~~~~~~~~~~~~~~~~

System dependencies
~~~~~~~~~~~~~~~~~~~

Environment Variables
---------------------

* ``EXODUS_DIR``: path to the Exodus library.

.. _Building TINerator:

Building TINerator
------------------

Setuptools builds both the main PyMesh module as well as all third party
dependencies. To build PyMesh::

    ./setup.py build


Build with CMake
~~~~~~~~~~~~~~~~

If you are familiar with C++ and CMake, there is an alternative way of building
PyMesh.  First compile and install all of the third party dependencies::

    cd $PYMESH_PATH/third_party
    ./build.py all

Third party dependencies will be installed in
``$PYMESH_PATH/python/pymesh/third_party`` directory.

It is recommended to build out of source, use the following commands setup building
environment::

    cd $PYMESH_PATH
    mkdir build
    cd build
    cmake ..

PyMesh consists of several modules.  To build all modules and their
corresponding unit tests::

    make
    make tests

PyMesh libraries are all located in ``$PYMESH_PATH/python/pymesh/lib``
directory.


Install PyMesh
~~~~~~~~~~~~~~

To install PyMesh in your system::

    ./setup.py install  # May require root privilege

Alternatively, one can install PyMesh locally::

    ./setup.py install --user


Post-installation check
~~~~~~~~~~~~~~~~~~~~~~~

To check PyMesh is installed correctly, one can run the unit tests::

    python -c "import pymesh; pymesh.test()"

Please make sure all unit tests are passed, and report any unit test
failures.
