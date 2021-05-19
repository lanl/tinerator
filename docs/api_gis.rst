GIS Module Functions
==========================

Loading files
-------------

.. autofunction:: tinerator.gis.load_raster
    :noindex:
.. autofunction:: tinerator.gis.load_shapefile
    :noindex:

Reprojection
------------

.. autofunction:: tinerator.gis.reproject_raster
.. autofunction:: tinerator.gis.reproject_geometry

Resample a Raster
-----------------

.. autofunction:: tinerator.gis.resample_raster

Clipping a Raster with Geometry
-------------------------------

.. autofunction:: tinerator.gis.clip_raster

Generate flowlines from a DEM using watershed delineation
---------------------------------------------------------

.. autofunction:: tinerator.gis.watershed_delineation

Converting a triangulation into a Geometry object
-------------------------------------------------

.. autofunction:: tinerator.gis.vectorize_triangulation

Rasterize Geometry
------------------

.. autofunction:: tinerator.gis.rasterize_geometry

Create a pyproj.CRS object
--------------------------

.. autofunction:: tinerator.gis.parse_crs

Compute the distance field from a Geometry object to all Raster cells
---------------------------------------------------------------------

.. autofunction:: tinerator.gis.distance_map
