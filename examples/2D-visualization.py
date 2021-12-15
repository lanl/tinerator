# -*- coding: utf-8 -*-
"""
2D Visualization
================

TINerator supports geo-referenced visualization of
rasters and vectors.

Available backends are:

* PlotlyJS
* Matplotlib


An arbitrary number of vector and raster objects
can be plotted together simultaneously.
"""

# %%
# This is a section header
# ------------------------
# This is the first section!

dem = tin.gis.load_raster()
dem.plot()

boundary = tin.gis.load_shapefile()
boundary.plot()

flowline = tin.gis.load_shapefile()
flowline.plot([boundary, dem])

# %%
# Alternately, the following syntax also works:
# 

tin.plot2d([dem, boundary, flowline])

# %%
# Source: ``2D-visualization.py``
# ------------------
#
# .. literalinclude:: 2D-visualization.py
#
#
