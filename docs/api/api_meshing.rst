Meshing Module Functions
========================

Loading a mesh from disk
------------------------

.. autofunction:: tinerator.meshing.load_mesh
    :noindex:

Mesh quality statistics
-----------------------

.. autofunction:: tinerator.meshing.triangle_quality
.. autofunction:: tinerator.meshing.triangle_area
.. autofunction:: tinerator.meshing.prism_volume

Extruding a surface mesh into a volume mesh
-------------------------------------------

.. autofunction:: tinerator.meshing.extrude_mesh

Generating a triangulation from a raster
----------------------------------------

.. autofunction:: tinerator.meshing.triangulate

Converting a triangulation into a Geometry object
-------------------------------------------------

.. autofunction:: tinerator.gis.vectorize_triangulation
    :noindex:

Estimate edge length size from triangle count
---------------------------------------------

.. autofunction:: tinerator.meshing.estimate_edge_lengths

Extracting a surface mesh and defining sets
-------------------------------------------

.. autofunction:: tinerator.meshing.SurfaceMesh
.. autofunction:: tinerator.meshing.NodeSet
.. autofunction:: tinerator.meshing.ElemSet
.. autofunction:: tinerator.meshing.SideSet