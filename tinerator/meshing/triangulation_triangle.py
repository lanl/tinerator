import triangle as tr
import numpy as np
from .mesh import Mesh, ElementType
from ..gis import map_elevation


def triangulation_triangle_uniform(raster, max_edge: float, quality: bool = True) -> Mesh:
    """
    Constructs a triangular mesh with roughly uniform edge lengths.
    Uses the `triangle` Delaunay triangulation software to generate
    mesh.
    """

    max_area = max_edge
    boundary = raster.get_boundary(distance=5., connect_ends=True)

    opts = {
        "q": quality,
        "p": True,
        "a": (round(max_area, 2)),
    }

    t = tr.triangulate(
        {
            "vertices": list(boundary.points[:, :2]),
            "segments": list(boundary.connectivity - 1),
        },
        # p enforces boundary connectivity,
        # q gives a quality mesh,
        # and aX is max edge length
        "pqa%f" % (round(max_area, 2)),
    )

    m = Mesh()
    m.nodes = np.hstack((t["vertices"], np.zeros((t["vertices"].shape[0], 1))))
    m.elements = t["triangles"] + 1
    m.element_type = ElementType.TRIANGLE

    m.nodes[:, 2] = map_elevation(dem, m.nodes)

    return m
    