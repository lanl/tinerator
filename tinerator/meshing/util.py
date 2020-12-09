from math import sqrt
from ..gis import Raster


def estimate_edge_lengths(raster: Raster, n_tris: int) -> float:
    """
    Estimates the edge lengths of `n_tris` amount of equilateral triangles
    that would fit into `raster`.
    """
    return sqrt(4.0 / sqrt(3.0) * (raster.area / float(n_tris)))
