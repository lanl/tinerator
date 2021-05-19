Installation
============

Docker
------

The easiest way of using TINerator is through
`docker <https://www.docker.com/>`_, where you can 
``pull`` a `pre-built image of TINerator <https://hub.docker.com/r/ees16/tinerator/>`_ from
dockerhub::

    $ docker run -it ees16/tinerator

This will launch a Jupyter Lab instance, ready to use.

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
