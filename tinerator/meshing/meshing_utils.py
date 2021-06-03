from math import sqrt
import numpy as np
from ..gis import Raster

def get_linestring_connectivity(
    nodes: np.array, closed: bool = True, clockwise: bool = True
) -> np.ndarray:
    """
    Internal function. Returns a connectivity array
    for an ordered array of nodes.
    Assumes that all points are ordered clockwise.
    """
    closed_conn = [(len(nodes), 1)] if closed else []
    connectivity = np.array(
        [(i, i + 1) for i in range(1, len(nodes))] + closed_conn, dtype=int
    )

    return connectivity


def estimate_edge_lengths(raster: Raster, n_tris: int) -> float:
    """
    Estimates the edge lengths of `n_tris` amount of equilateral triangles
    that would fit into `raster`.
    """
    return sqrt(4.0 / sqrt(3.0) * (raster.area / float(n_tris)))
