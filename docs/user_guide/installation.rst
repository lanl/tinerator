.. _installation:

Installation
============

Docker
------

The easiest way of using TINerator is through
`Docker <https://www.docker.com/>`_, where you can
``pull`` a `pre-built image of TINerator <https://hub.docker.com/r/ees16/tinerator/>`_ from
DockerHub::

    $ docker run --publish 8899:8899 --volume $(pwd):/home/feynman/work ees16/tinerator:latest

When run, the Docker container initializes a Jupyter Lab instance.

You will see some output similar to this:

.. code:: text

    [I 2021-05-20 22:48:24.778 LabApp] JupyterLab application directory is /usr/local/share/jupyter/lab
    [I 2021-05-20 22:48:24.782 ServerApp] jupyterlab | extension was successfully loaded.
    [I 2021-05-20 22:48:24.783 ServerApp] Serving notebooks from local directory: /tinerator/playground
    [I 2021-05-20 22:48:24.783 ServerApp] Jupyter Server 1.8.0 is running at:
    [I 2021-05-20 22:48:24.783 ServerApp] http://dae7d8d45c90:8899/lab
    [I 2021-05-20 22:48:24.783 ServerApp]     http://127.0.0.1:8899/lab
    [I 2021-05-20 22:48:24.784 ServerApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).

Open an internet browser on your machine and navigate to `http://127.0.0.1:8899/lab <http://127.0.0.1:8899/lab>`_. If all is successful,
the Jupyter Lab logo should appear, along with a side panel showing `docs` and `examples` folders.

Clarifying the Docker command
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The above `docker run` command requires a lot of flags. They are:

- ``--publish 8899:8899``: Maps the network port 8888 within the Docker container
  to the network port 8899 on your computer. This is what allows you to
  access the Jupyter Lab that is contained within Docker from your internet
  browser.
- ``--volume $(pwd):/home/feynman/work``: Docker cannot view files or folders on your machine
  unless it's explicitly allowed to via this command. This takes your current working directory
  ``$(pwd)`` and makes it visible from within the Docker container, at the location ``/home/feynman/work``.

Interfacing with Docker via Bash instead of Jupyter Lab
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can bypass Jupyter Lab and use TINerator through a Bash shell, if desired, by appending the Docker command with ``/bin/bash/``:

    $ docker run -it --volume $(pwd):/tinerator/work ees16/tinerator:latest /bin/bash

Note that the ``--publish`` flag is not needed if bypassing Jupyter, as there is no need for the host machine
to communicate with Docker over networking ports.
- ``-it``: short for ``--interactive`` + ``--tty``

PyPI
----

To install through [PyPI](https://pypi.org/project/tinerator/):

.. code:: bash

    $ pip install tinerator

Conda
-----

.. code:: bash

    $ conda install -c conda-forge tinerator

Through GitHub
--------------

The source code can be checked out from
`GitHub <https://github.com/lanl/tinerator>`_::

    $ git clone https://github.com/lanl/tinerator.git

Then,

    $ cd tinerator/
    $ python -m pip install ".[all]"

Troubleshooting
===============

The most common point of failure is installing GDAL.
The Conda Forge recipe is the easiest way to get GDAL
working:

.. code:: bash

    $ conda install gdal

If you have GDAL installed independently (i.e., through
``brew`` or ``apt``), you may install Python bindings through
PyPI:

.. code:: bash

    $ pip install GDAL==`gdal-config --version`
