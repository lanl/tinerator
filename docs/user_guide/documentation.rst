.. _documentation:

Documentation
=============

Building Dependencies
---------------------

To build documentation dependencies, run:

.. code-block:: bash

    $ python -m pip install ".[docs]"

This will install all dependencies under the ``docs``
portion of ``[options.extras_require]`` in ``setup.cfg``.

At the time of writing, these are:

.. code-block:: yaml

    [options.extras_require]
    docs =
        sphinx
        furo
        sphinx-autodocgen
        sphinx-inline-tabs
        sphinx-copybutton
        sphinx-gallery

Building HTML Documentation
---------------------------

To build the documentation, run:

.. code-block:: bash

    $ make -C docs/ html

The HTML documentation files will be in ``docs/_build/``.

Building PDF Documentation
--------------------------

Similar to building HTML docs, just run:

.. code-block:: bash

    $ make -C docs/ pdf

Again, the compiled files will be in ``docs/_build/``.