import triangle as tr
import numpy as np
from .mesh import Mesh, ElementType

def uniform_mesh(dem, max_edge):
    vertices, connectivity = dem.get_boundary(5., connect_ends=True)

    t = tr.triangulate(
        {
            'vertices': list(vertices[:,:2]),
            'segments': list(connectivity - 1),
        }, 
        # p enforces boundary connectivity, 
        # q gives a quality mesh, 
        # and aX is max edge length
        'pqa%f' % (round(max_edge,2))
    )

    m = Mesh()
    m.nodes = np.hstack(
        (
            t['vertices'],
            np.zeros((t['vertices'].shape[0],1))
        )
    )
    m.elements = t['triangles'] + 1
    m.element_type = ElementType.TRIANGLE

    return m

'''
TODO:
        self.boundary = util.xyVectorToProjection(self.boundary,
                                                  self.cell_size,
                                                  self.xll_corner,
                                                  self.yll_corner,
                                                  self.nrows)
'''