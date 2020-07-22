import numpy as np
import sys
sys.path.insert(0,'/Users/livingston/playground/tinerator/tinerator-core')
import tinerator as tin
from tinerator.meshing import Mesh, ElementType

def init_surf_mesh_tri():
    '''Simple triangle surface mesh'''

    nodes = np.array([
        [1.000000000000E+01, 1.000000000000E+01, 1.000000000000E+01],
        [2.000000000000E+01, 1.000000000000E+01, 1.000000000000E+01],
        [2.000000000000E+01, 2.000000000000E+01, 1.000000000000E+01],
        [1.000000000000E+01, 2.000000000000E+01, 1.000000000000E+01],
        [1.500000000000E+01, 1.500000000000E+01, 1.000000000000E+01],
    ], dtype=float)

    connectivity = np.array([
        [1, 5, 4],
        [5, 3, 4],
        [1, 2, 5],
        [5, 2, 3],
    ], dtype=int)

    m = Mesh()
    m.nodes = nodes
    m.elements = connectivity
    m.element_type = ElementType.TRIANGLE

    return m