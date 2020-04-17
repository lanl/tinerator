import numpy as np
from enum import Enum, auto

class ElementType(Enum):
    TRIANGLE = auto()
    QUAD = auto()
    PRISM = auto()
    HEX = auto()

class Mesh:
    def __init__(self, name = 'mesh object', nodes = None, elements = None, etype = None):
        self.name = name
        self.nodes = nodes
        self.elements = elements
        self.element_type = etype
        self.metadata = {}
        self.attributes = {
            'node': {}, 
            'cell': {}
        }
    
    def __repr__(self):
        return "Mesh<name: \"{}\", nodes: {}, elements<{}>: {}>".format(
            self.name,
            self.n_nodes,
            self.element_type,
            self.n_elements
        )
    
    @property
    def x(self):
        '''X vector of mesh nodes'''
        if self.nodes is not None:
            return self.nodes[:,0]
        return None

    @property
    def y(self):
        '''Y vector of mesh nodes'''
        if self.nodes is not None:
            return self.nodes[:,1]
        return None

    @property
    def z(self):
        '''Z vector of mesh nodes'''
        if self.nodes is not None:
            return self.nodes[:,2]
        return None
    
    @property
    def n_nodes(self):
        '''Number of nodes in mesh'''
        if self.nodes is not None:
            return self.nodes.shape[0]
        return None
    
    @property
    def n_elements(self):
        '''Number of elements in mesh'''
        if self.elements is not None:
            return self.elements.shape[0]
        return None

    @property
    def centroid(self):
        '''Mesh centroid'''
        if self.nodes is not None:
            return (np.mean(self.x),np.mean(self.y),np.mean(self.z))
        return None
    
    @property
    def extent(self):
        '''
        Returns the extent of the mesh: 
            [ (x_min, x_max), (y_min, y_max), (z_min, z_max) ]
        '''

        if self.nodes is None:
            return None
        
        ex = []
        for i in range(3):
            vector = self.nodes[:,i]
            ex.append((np.min(vector),np.max(vector)))

        return ex