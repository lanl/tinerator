# -*- coding: utf-8 -*-
"""
Meshing from GIS Data
=====================


"""

import tinerator as tin

new_mexico = tin.examples.NewMexico()

dem = new_mexico.dem
flowline = new_mexico.flowline

# %%
# Mesh Triangulation
# ------------------
#

mesh = tin.meshing.triangulate(
    dem,
    min_edge_length=0.01,
    max_edge_length=0.05,
    refinement_feature=flowline,
    method="jigsaw",
    scaling_type="relative",
)

# %%
# Source: ``meshing.py``
# ------------------
#
# .. literalinclude:: meshing.py
#
#
