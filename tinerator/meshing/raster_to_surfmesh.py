import numpy as np
from .mesh import Mesh
from .meshing_types import ElementType


def raster_to_quadmesh(
    raster: np.ndarray, extent: tuple, sampling_window: tuple = (1, 1)
) -> Mesh:
    m = Mesh()
    m.nodes = None
    m.elements = None
    m.element_type = ElementType.QUAD
    return m


def raster_to_triplane():
    pass
