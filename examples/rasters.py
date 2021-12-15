# -*- coding: utf-8 -*-
"""
Rasters
=======


"""

dem.fill_depressions()

# %%
# Clipping a Raster with a Polygon
# --------------------------------

dem = tin.gis.clip_raster(dem, boundary)

# %%
# Reprojecting Rasters
# --------------------

dem = tin.gis.reproject_raster(dem, 3456)

# %%
# Exporting Rasters
# -----------------

dem.save("dem-modified.tif")

# %%
# Source: ``rasters.py``
# ------------------
#
# .. literalinclude:: rasters.py
#
#
