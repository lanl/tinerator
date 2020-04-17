import numpy as np
from copy import deepcopy
from enum import Enum, auto
from mesh import Mesh, ElementType

V = '1'

class LayerType(Enum):
    UNIFORM = auto()
    PROPORTIONAL = auto()

class Layer:
    '''
    Sublayering object - when used with other Layer objects,
    can construct a fully layered, volumetric mesh.
    '''
    def __init__(self, sl_type, depth:float, nlayers:int, matids:list = None, data = None):

        if matids is not None:
            assert len(matids) == nlayers, 'Each layer required a material ID value'

        self.sl_type = sl_type
        self.depth = depth
        self.nlayers = nlayers
        self.matids = matids
        self.data = data
        self.layer = None
    
    def __repr__(self):
        return "Layer<{}, {}, {}>".format(self.sl_type,self.nlayers,self.data)

def proportional_sublayering() -> Layer:
    pass

def uniform_sublayering(depth: float, nsublayers: int, matids: list = None) -> Layer:
    return Layer(
        LayerType.UNIFORM,
        depth,
        nsublayers,
        data = [1]*nsublayers,
        matids = matids,
    )
    

def stack(surfmesh, layers:list):
    '''
    Extrudes and layers a surface mesh into a volumetric mesh, given
    the contraints in the `layers` object.
    '''
    
    if surfmesh.element_type != ElementType.TRIANGLE:
        raise ValueError("Mesh must be triangular; is actually: {}".format(surfmesh.element_type))

    if not all([isinstance(x,Layer) for x in layers]):
        raise ValueError('`layers` must be a list of Layers')

    vol_mesh = Mesh(name='stacked mesh', etype=ElementType.TRIANGLE)#etype=mesh.ElementType.PRISM)

    layer = layers[0]

    top_layer = deepcopy(surfmesh)
    vol_mesh.nodes = top_layer.nodes
    vol_mesh.elements = top_layer.elements

    z_abs = np.mean(top_layer.z) - np.abs(layer.depth)

    bottom_layer = deepcopy(surfmesh)
    bottom_layer.nodes[:,2] = np.full((bottom_layer.n_nodes,),z_abs)

    n_layer_cells = layer.nlayers
    n_layer_planes = n_layer_cells + 1

    middle_layers = []
    for i in range(n_layer_planes - 2):
        layer_plane = deepcopy(surfmesh)
        # Below, we are just doing a basic linear interpolation function between the 
        # top and the bottom layers.
        layer_plane.nodes[:,2] = ((i+1)*((top_layer.z-bottom_layer.z)/(n_layer_planes-1))+bottom_layer.z)
        middle_layers.append(layer_plane)

    all_layers = [*middle_layers,bottom_layer]

    for layer in all_layers:
        layer.elements += vol_mesh.n_nodes
        vol_mesh.nodes = np.vstack((vol_mesh.nodes,layer.nodes))
        vol_mesh.elements = np.vstack((vol_mesh.elements,layer.elements))

    return vol_mesh

    
















def proportional_sublayering(z_delta:float, sub_thick:list, matids=None):
    '''
    '''
    nsublayer = len(sub_thick)
    unit = z_delta / sum(sub_thick)

    top = 0

    depths = [top]
    for i in range(nsublayer):
        depths.append(depths[i] - sub_thick[i]*unit)
    
    if matids is None:
        matids = [i+1 for i in range(nsublayer)]

    self._stacked_mesh(depths)