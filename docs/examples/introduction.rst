.. _introduction:

Basic Usage
===========

Importing TINerator
-------------------

For brevity's sake, TINerator is usually imported as:

    >>> import tinerator as tin

Primary Data Structures
-----------------------

Within TINerator, there are three main classes:

* :obj:`tinerator.gis.Raster` - A representation of a GIS raster object
* :obj:`tinerator.gis.Geometry` - A representation of a GIS shapefile object
* :obj:`tinerator.meshing.Mesh` - The mesh class; typically with triangular or prism elements

The core of TINerator is the translation from ``Raster -> Mesh`` or ``Raster + Geometry -> Mesh``.

Loading GIS data
----------------

A raster can be loaded from disk using:

    >>> dem = tin.gis.load_raster("dem.tif")

All formats `supported by GDAL <https://gdal.org/drivers/raster/index.html>`_ can be read.

Shapefiles are loaded via:

    >>> geom = tin.gis.load_shapefile("flowline.shp")

TINerator uses `Fiona <https://fiona.readthedocs.io>`_ as the file reader, and supports all
file formats that Fiona does.

Visualizing GIS data
--------------------

Both `Raster` and `Geometry` have a `plot()` method:

    >>> dem.plot()
    >>> geom.plot()

Multiple Raster and Geometry objects can be plotted at once, by using the layers
command:

    >>> dem.plot(layers=[geom, raster2, geom2, geom3])

Saving GIS objects
------------------

Both Rasters and Geometry objects can be saved with the `save()` method:

    >>> dem.save("my_dem.tif")
    >>> geom.save("my_shapefile.shp")

This can be convenient for saving objects after performing operations on them (like clipping,
reprojecting, etc.).

Reprojecting GIS objects
------------------------

Both Raster and Geometry objects can be reprojected with a given coordinate reference system (CRS).
The CRS can take the form of an EPSG code, a WKT string, a Pyproj4 string, or a pyproj4 dict.

    >>> new_dem = tin.gis.reproject_raster(dem, "epsg:1234")
    >>> new_geom = tin.gis.reproject_geometry(geom, "+proj=latlon")

Under the hood, TINerator uses `pyproj <https://pyproj4.github.io/pyproj/stable/examples.html>`_ for
parsing and managing CRS'.

Clipping a Raster with a Geometry object
----------------------------------------

If the Geometry object is a type of ``Polygon``, you can clip a Raster object with it.
Everything in the Raster contained within the Geometry polygon will be kept, and everything else
will be discarded or set to `no_data_value`.

    >>> boundary = tin.gis.load_shapefile("watershed_boundary.shp")
    >>> print(boundary.geometry_type)
    MultiPolygonZ
    >>> dem = tin.gis.load_raster("dem.tif")
    >>> clipped_dem = tin.gis.clip_raster(dem, boundary)
    >>> clipped_dem.plot()

Meshing
=======

To generate a surface mesh from a volume mesh:

    >>> surface_mesh = volume_mesh.surface_mesh()
    >>> surface_mesh.view()
    >>> suface_mesh.save("surf.vtk")

Top, bottom, and sides point and side sets are auto-generated:

    >>> top_faces = surface_mesh.top_faces
    >>> bottom_faces = surface_mesh.bottom_faces
    >>> side_faces = surface_mesh.side_faces
    >>> top_nodes = surface_mesh.top_nodes

You can view any set object with the :obj:`tinerator.meshing.Mesh.view` function:

    >>> vol_mesh.view(sets=[top_faces, top_nodes, bottom_faces])

And export to ExodusII format in the same way:

    >>> vol_mesh.save("full_mesh.exo", sets=[top_faces, bottom_faces, top_nodes])