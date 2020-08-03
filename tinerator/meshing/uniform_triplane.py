import triangle as tr
import numpy as np
from .mesh import Mesh, ElementType
from ..gis import map_elevation


def get_uniform_triplane(dem, max_edge: float) -> Mesh:
    """
    Constructs a triangular mesh with roughly uniform edge lengths.
    """
    max_area = max_edge
    vertices, connectivity = dem.get_boundary(5.0, connect_ends=True)

    t = tr.triangulate(
        {
            "vertices": list(vertices[:, :2]),
            "segments": list(connectivity - 1),
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

    z_values = map_elevation(dem, m.nodes)
    m.nodes[:, 2] = z_values

    return m
