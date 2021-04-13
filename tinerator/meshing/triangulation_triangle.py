import triangle as tr
import numpy as np
import random
import os
import tempfile
from .mesh import load_mesh, Mesh, ElementType
from ..gis import map_elevation, Raster, Shape, distance_map
from ..logging import log, warn, debug

"""
        raster,
        raster_boundary=boundary,
        refinement_feature=refinement_feature,
        min_edge_length=min_edge_length,
        max_edge_length=max_edge_length,
        scaling_type=scaling_type,
        **kwargs,
    )
"""


def triangulation_triangle(
    raster: Raster,
    raster_boundary: Shape = None,
    min_edge_length: float = None,
    max_edge_length: float = None,
    refinement_feature: Shape = None,
    scaling_type: str = "relative",
):
    """
    Constructs a triangular mesh with the Triangle
    meshing software.
    """

    vertices = raster_boundary.points[:, :2]
    segments = raster_boundary.connectivity - 1

    vertex_markers = np.array([[2] for _ in range(len(vertices))], dtype=np.int32)
    segment_markers = np.array([[2] for _ in range(len(segments))], dtype=np.int32)

    watershed = {
        "vertices": vertices,
        "vertex_markers": vertex_markers,
        "segments": segments.astype(np.int32),
        "segment_markers": segment_markers,
    }

    # {
    #    'vertices': [],
    #    'vertex_markers':
    # }

    # https://github.com/inducer/meshpy/blob/main/doc/conf.py#L99
    # https://documen.tician.de/meshpy/_modules/meshpy/triangle.html#build

    quality_meshing = True
    verbose = True
    attributes = False
    max_volume = False
    refinement_func = None
    min_angle = None

    opts = "pzj"

    if quality_meshing:
        if min_angle is not None:
            opts += "q%f" % min_angle
        else:
            opts += "q"

    # if mesh_order is not None:
    #    opts += "o%d" % mesh_order

    if verbose:
        opts += "VV"
    else:
        opts += "Q"

    if attributes:
        opts += "A"

    if max_volume:
        opts += "a%0.20f" % max_volume

    if refinement_func is not None:
        opts += "u"

    from matplotlib import pyplot as plt

    plt.scatter(vertices[:, 0], vertices[:, 1])
    plt.show()
    import ipdb

    ipdb.set_trace()
    opts = "p"
    t = tr.triangulate(watershed, opts)
    # tr.compare(plt, watershed, t)
    # plt.show()

    # import ipdb; ipdb.set_trace()

    m = Mesh()
    m.nodes = np.hstack((t["vertices"], np.zeros((t["vertices"].shape[0], 1))))
    m.elements = t["triangles"] + 1
    m.element_type = ElementType.TRIANGLE

    # m.nodes[:, 2] = map_elevation(raster, m.nodes)

    return m

    # mesh = MeshInfo()
    # internals.triangulate(opts, mesh_info, mesh, MeshInfo(), refinement_func)


# def triangulation_triangle_uniform(
#    raster, max_edge: float, quality: bool = True
# ):
#    """
#    Constructs a triangular mesh with roughly uniform edge lengths.
#    Uses the `triangle` Delaunay triangulation software to generate
#    mesh.
#    """#

#    max_area = max_edge
#    boundary = raster.get_boundary(distance=5.0, connect_ends=True)#

#    # opts = {"q": quality, "p": True, "a": (round(max_area, 2))}#

#    t = tr.triangulate(
#        {
#            "vertices": list(boundary.points[:, :2]),
#            "segments": list(boundary.connectivity - 1),
#        },
#        # p enforces boundary connectivity,
#        # q gives a quality mesh,
#        # and aX is max edge length
#        "pqa%f" % (round(max_area, 2)),
#    )#

#    m = Mesh()
#    m.nodes = np.hstack((t["vertices"], np.zeros((t["vertices"].shape[0], 1))))
#    m.elements = t["triangles"] + 1
#    m.element_type = ElementType.TRIANGLE#

#    m.nodes[:, 2] = map_elevation(raster, m.nodes)#

#    return m
