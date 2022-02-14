# -*- coding: utf-8 -*-
"""
File Formats & Import
=====================

TINerator supports many different file formats.

+--------------------+----------------------------+--------------------+
| Rasters            | Vectors                    | Meshes             |
+====================+============================+====================+
| GeoTIFF (``.tif``) | ESRI Shapefiles (``.shp``) | AVS-UCD (``.inp``) |
+--------------------+----------------------------+--------------------+
|                    |                            |                    |
+--------------------+----------------------------+--------------------+
|                    |                            |                    |
+--------------------+----------------------------+--------------------+

Below, we will go over how to import files in these formats.

"""
import os
import tinerator as tin

print(f"TINerator version: {tin.__version__}")

# %%
# Rasters
# -------
# Rasters are :math:`NxM` matrices where the data
# represents something like elevation, vegetation maps,
# or soil type. TINerator can read both binary and ASCII
# rasters.
#
# The base command to read rasters is
# ``tin.gis.load_raster``.
#
# .. autofunction:: tinerator.gis.load_raster
#
# Here's an example of loading a GeoTIFF:
#

example_raster_fname = os.path.join(
    tin.examples.DATA_DIR, "new_mexico/rasters/USGS_NED_13_n37w108_Clipped.tif"
)

dem = tin.gis.load_raster(example_raster_fname)
print(dem)

# %%
# Vectors
# -------
# Vector objects -- typically just called "shapefiles"
# in GIS parlance -- are representations of 2D geometrical
# primitives. Commonly, these represent watershed boundaries,
# flowlines, or regions of interest.
#
# The base command to read vectors is
# ``tin.gis.load_shapefile``.
#
# .. autofunction:: tinerator.gis.load_shapefile
#
# Here's an example of loading an ESRI Shapefile:
#
example_vector_fname = os.path.join(
    tin.examples.DATA_DIR, "new_mexico/shapefiles/WBDHU12/WBDHU12.shp"
)
watershed_boundary = tin.gis.load_shapefile(example_vector_fname)
watershed_boundary.plot()

# %%
# Source: ``file-import.py``
# ------------------
#
# .. literalinclude:: file-import.py
#
#
