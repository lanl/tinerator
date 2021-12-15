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

import tinerator as tin

new_mexico = tin.examples.NewMexico()

# %%
# Plotting individual objects
# ---------------------------
# Individual objects can be plotted
# with ``name.plot()

dem = tin.gis.load_raster(new_mexico.dem)
dem.plot()

boundary = tin.gis.load_shapefile(new_mexico.watershed_boundary)
boundary.plot()

flowline = tin.gis.load_shapefile(new_mexico.flowline)
flowline.plot([boundary, dem])

# %%
# Alternately, the following syntax also works:
# 

tin.plot2d([dem, boundary, flowline])

# %%
# Saving Plots
# ------------
# Plots can be saved to static images, and
# in some cases, to HTML as well:

boundary.plot(outfile="boundary_plot.png")
boundary.plot(outfile="boundary_plot.pdf")
boundary.plot([dem, flowline], outfile="boundary_plot.html")

# %%
# Source: ``2D-visualization.py``
# ------------------
#
# .. literalinclude:: 2D-visualization.py
#
#
