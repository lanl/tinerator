import meshpy.triangle
import numpy as np
import random
import os
import tempfile
from .mesh import load_mesh, Mesh, ElementType
from .meshing_utils import get_linestring_connectivity
from ..gis import map_elevation, Raster, Geometry, distance_map, unproject_vector
from ..logging import log, warn, debug


def triangulation_meshpy(
    raster: Raster,
    raster_boundary: Geometry = None,
    min_edge_length: float = None,
    max_edge_length: float = None,
    refinement_feature: Geometry = None,
    scaling_type: str = "relative",
    verbose: bool = False,
    min_triangle_area: float = None,
    max_triangle_area: float = None,
    **kwargs,
):
    """
    Constructs a triangular mesh with the MeshPy
    meshing software.
    """

    def ideal_triangle_area(edge_length):
        """Returns area of equilateral triangle."""
        return (3 ** 0.5 / 4.0) * edge_length ** 2

    if scaling_type == "relative":
        dX = np.abs(raster.extent[2] - raster.extent[0])
        dY = np.abs(raster.extent[3] - raster.extent[1])
        coeff = (dX + dY) / 2.0
    else:
        coeff = 1.0

    if min_edge_length is None:
        min_area = None
    else:
        min_area = ideal_triangle_area(min_edge_length * coeff)

    if max_edge_length is None:
        max_area = None
    else:
        max_area = ideal_triangle_area(max_edge_length * coeff)

    if max_triangle_area is not None:
        max_area = max_triangle_area
    if min_triangle_area is not None:
        min_area = min_triangle_area

    if refinement_feature is None:
        TARGET_AREA = max_area if max_area is not None else min_area

        def refinement_callback(vertices, area):
            return area > TARGET_AREA

    else:
        dmap = distance_map(
            raster,
            refinement_feature,
            min_dist=min_area,
            max_dist=max_area,
        )

        def refinement_callback(vertices, area):
            idx = unproject_vector(np.array(vertices), raster)
            idx = np.round(np.mean(idx, axis=0)).astype(int)
            target_area = dmap[idx[1], idx[0]]

            return area > target_area

    kwargs["refinement_func"] = refinement_callback

    vertices = np.array(raster_boundary.shapes[0].coords[:])[:, :2]
    segments = get_linestring_connectivity(vertices, closed=True, clockwise=True)
    segments -= 1

    mesh_info = meshpy.triangle.MeshInfo()
    mesh_info.set_points(vertices)  # boundary points
    mesh_info.set_facets(segments)  # connectivity
    mesh = meshpy.triangle.build(mesh_info, verbose=verbose, **kwargs)

    mesh_points = np.array(mesh.points).astype(float)
    mesh_points = np.hstack([mesh_points, np.zeros((mesh_points.shape[0], 1))])
    mesh_tris = np.array(mesh.elements).astype(int) + 1

    m = Mesh(nodes=mesh_points, elements=mesh_tris, etype=ElementType.TRIANGLE)
    m.nodes[:, 2] = map_elevation(raster, m.nodes)
    mesh.crs = raster_boundary.crs

    return m
