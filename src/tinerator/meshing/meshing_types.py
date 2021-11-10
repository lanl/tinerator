import numpy as np
from enum import Enum, auto
import vtk


class ElementType(Enum):
    TRIANGLE = auto()
    QUAD = auto()
    PRISM = auto()
    HEX = auto()
    POLYGON = auto()


# Standard primitives in ExodusII format
class ExodusElements(Enum):
    # ExodusII API: https://gsjaardema.github.io/seacas-docs/exodusII-new.pdf
    # pp. 16
    LINE = "Bar2"
    QUAD = "Quad4"
    TRIANGLE = "Tri3"
    TETRA = "Tet4"  # Tetrahedron
    WEDGE = "Wedge6"  # Prism
    HEX = "Hex8"


_ex = ExodusElements

# Mapping between (sorted low-high) face
# connectivity and ExodusII face type
EXODUS_CELL_FACES = {
    # ExodusII API: https://gsjaardema.github.io/seacas-docs/exodusII-new.pdf
    # pp. 28
    _ex.QUAD: {
        (1, 2): 1,
        (2, 3): 2,
        (3, 4): 3,
        (1, 4): 4,
    },
    _ex.TRIANGLE: {
        (1, 2): 1,
        (2, 3): 2,
        (1, 3): 3,
    },
    _ex.TETRA: {
        (1, 2, 4): 1,
        (2, 3, 4): 2,
        (1, 3, 4): 3,
        (1, 2, 3): 4,
    },
    _ex.WEDGE: {
        (1, 2, 4, 5): 1,
        (2, 3, 5, 6): 2,
        (1, 3, 4, 6): 3,
        (1, 2, 3): 4,
        (4, 5, 6): 5,
    },
    _ex.HEX: {
        (1, 2, 5, 6): 1,
        (2, 3, 6, 7): 2,
        (3, 4, 7, 8): 3,
        (1, 4, 5, 8): 4,
        (1, 2, 3, 4): 5,
        (5, 6, 7, 8): 6,
    },
}

AVS_TYPE_MAPPING = {"tri": ElementType.TRIANGLE, "prism": ElementType.PRISM}

# This maps what faces can be extracted from specific cells
# Schema:
# face_type = VTK_CELL_FACES[parent_cell_type][num_nodes_in_face]
VTK_CELL_FACES = {
    vtk.VTK_WEDGE: {
        3: vtk.VTK_TRIANGLE,
        4: vtk.VTK_QUAD,
    },
    vtk.VTK_TRIANGLE: {
        2: vtk.VTK_LINE,
    },
}


# Mapping between VTK and ExodusII cell types
# Note that **connectivity** is not necessarily
# the same and will need to be altered!
VTK_CELL_TYPES_TO_EXODUS = {
    vtk.VTK_LINE: _ex.LINE,
    vtk.VTK_QUAD: _ex.QUAD,
    vtk.VTK_TRIANGLE: _ex.TRIANGLE,
    vtk.VTK_TETRA: _ex.TETRA,
    vtk.VTK_WEDGE: _ex.WEDGE,
    vtk.VTK_HEXAHEDRON: _ex.HEX,
}
