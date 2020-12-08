import numpy as np
from copy import copy, deepcopy
from enum import Enum, auto
from .mesh import Mesh, StackedMesh, ElementType

DEBUG = True

# TODO: is this unnecessary?
class LayerType(Enum):
    UNIFORM = auto()
    PROPORTIONAL = auto()


# TRANSLATE
# FLATTEN


class Layer:
    """
    Sublayering object - when used with other Layer objects,
    can construct a fully layered, volumetric mesh.
    """

    def __init__(
        self,
        sl_type,
        depth: float,
        nlayers: int,
        matids: list = None,
        data=None,
        relative_z=True,
    ):

        if matids is not None:
            assert (
                len(matids) == nlayers
            ), "Each layer required a material ID value"

        self.sl_type = sl_type
        self.depth = depth
        self.nlayers = nlayers
        self.matids = matids
        self.data = data
        self.layer = None
        self.relative_z = relative_z

    def __repr__(self):
        return "Layer<{}, {}, {}>".format(
            self.sl_type, self.nlayers, self.data
        )


def proportional_sublayering(
    depth: float, sub_thick: list, matids: list = None, relative_z: bool = True
) -> Layer:
    return Layer(
        LayerType.PROPORTIONAL,
        depth,
        len(sub_thick),
        data=sub_thick,
        matids=matids,
        relative_z=relative_z,
    )


def uniform_sublayering(
    depth: float, nsublayers: int, matids: list = None, relative_z: bool = True
) -> Layer:
    return Layer(
        LayerType.UNIFORM,
        depth,
        nsublayers,
        data=[1] * nsublayers,
        matids=matids,
        relative_z=relative_z,
    )


def stack(surfmesh: Mesh, layers: list, matids: list = None) -> Mesh:
    """
    Extrudes and layers a surface mesh into a volumetric mesh, given
    the contraints in the `layers` object.
    """

    if surfmesh.element_type != ElementType.TRIANGLE:
        raise ValueError(
            "Mesh must be triangular; is actually: {}".format(
                surfmesh.element_type
            )
        )

    if isinstance(layers, Layer):
        layers = [layers]

    layers = list(layers)

    if not all([isinstance(x, Layer) for x in layers]):
        raise ValueError("`layers` must be a list of Layers")

    top_layer = deepcopy(surfmesh)

    # Initialize the volumetric mesh
    vol_mesh = StackedMesh(name="stacked mesh", etype=ElementType.PRISM)
    vol_mesh.nodes = top_layer.nodes
    vol_mesh.elements = top_layer.elements

    # This replaces the above deprecation.
    vol_mesh._nodes_per_layer = vol_mesh.n_nodes
    vol_mesh._elems_per_layer = vol_mesh.n_elements

    mat_ids = []
    total_layers = 0

    for (i, layer) in enumerate(layers):

        total_layers += layer.nlayers
        n_layer_planes = layer.nlayers + 1

        if layer.relative_z:
            z_abs = np.min(top_layer.z) - np.abs(layer.depth)
        else:
            z_abs = layer.depth

        if layer.matids is None:
            mat_ids.extend([1] * layer.nlayers)
        else:
            mat_ids.extend(layer.matids)

        bottom_layer = deepcopy(surfmesh)  # should this be top_layer?

        # Below is where we set the bottom layer to be flat - this could (and should)
        # have the option to be either a different topology or the same topology. Or flat!
        bottom_layer.nodes[:, 2] = np.full((bottom_layer.n_nodes,), z_abs)

        middle_layers = []

        # Create and set middle (sandwich ingredients) layers (if any)
        for j in range(n_layer_planes - 2):
            layer_plane = deepcopy(surfmesh)

            # Below, we are just setting layer Z values via a basic linear interpolation function.
            k = sum(layer.data[: j + 1]) / sum(layer.data)
            layer_plane.nodes[:, 2] = (
                float(k) * (top_layer.z - bottom_layer.z) + bottom_layer.z
            )

            middle_layers.append(layer_plane)

        # Invert sandwich layers so they are in the proper order
        if middle_layers:
            middle_layers = middle_layers[::-1]

        # It's important that the top layer isn't added here. Duplication of nodes.
        all_layers = [*middle_layers, bottom_layer]

        # Append all nodes and elements from layers into the volumetric mesh
        for (j, l) in enumerate(all_layers):

            if DEBUG:
                l.save("DEBUG_layer_%d%d.inp" % (i, j))

            l.elements += vol_mesh.n_nodes
            # vol_mesh.nodes = np.vstack((vol_mesh.nodes, l.nodes))
            vol_mesh.nodes = np.vstack((l.nodes, vol_mesh.nodes))
            vol_mesh.elements = np.vstack((vol_mesh.elements, l.elements))

        top_layer = deepcopy(bottom_layer)

    vol_mesh._num_layers = total_layers

    # Join 'stacked' triangles into prisms
    elems_per_layer = vol_mesh._elems_per_layer
    n_prisms = vol_mesh.n_elements - elems_per_layer
    prisms = np.zeros((n_prisms, 6), dtype=int)

    # Generate prisms from pairwise triangles
    for i in range(n_prisms):
        prisms[i] = np.hstack(
            (vol_mesh.elements[i + elems_per_layer], vol_mesh.elements[i])
        )

    # Swap columns 1 and 2 - reversing triangle connectivity
    swap_a = copy(prisms[:, 1])
    swap_b = copy(prisms[:, 2])

    prisms[:, 1] = swap_b
    prisms[:, 2] = swap_a

    # Swap columns 4 and 5 - reversing triangle connectivity
    swap_a = copy(prisms[:, 4])
    swap_b = copy(prisms[:, 5])

    prisms[:, 4] = swap_b
    prisms[:, 5] = swap_a

    vol_mesh.elements = prisms

    # This should probably be in its own method
    vol_mesh.add_attribute(
        "material_id",
        np.repeat(np.array(mat_ids[::-1], dtype=int), elems_per_layer),
        attrb_type="cell",
    )

    # Each element in a layer will get its own integer value
    # This can then be used to 'mask' cells belonging to different layers
    vol_mesh._cell_layer_ids = np.repeat(
        np.array(list(range(total_layers)), dtype=int), elems_per_layer
    )

    return vol_mesh

def extrude_surface(surfmesh: Mesh, layers: list, matids: list = None) -> Mesh:
    '''
    Simple multilayer extrusion.
    '''

    layer_objs = []

    for (i, layer) in enumerate(layers):
        # Allow for a naive version of layering: just pass in the layer depths
        if isinstance(layer, (int, float)):
            depth = layer
            subdivisions = 1
        elif isinstance(layer, list):
            depth, subdivisions = layer

        if matids is None:
            matid_i = [i + 1] * subdivisions
        else:
            matid = matids[i]

        layer_objs.append(uniform_sublayering(depth, subdivisions, matids = matid_i, relative_z = True))

    return stack(surfmesh, layer_objs)