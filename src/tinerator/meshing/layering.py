import numpy as np
from copy import copy, deepcopy
from enum import Enum, auto
from typing import List
from .mesh import Mesh, StackedMesh
from .meshing_types import ElementType

# TODO: what happens when the nodes on surface layer intersects a
# flat sublayer? I.e., the prisms generated would have 0 volume.
# Nodes should be de-duped or prisms should have non-zero volume (bad).


class LayerType(Enum):
    UNIFORM = auto()
    PROPORTIONAL = auto()
    TRANSLATED = auto()


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
            assert len(matids) == nlayers, "Each layer required a material ID value"

        self.sl_type = sl_type
        self.depth = depth
        self.nlayers = nlayers
        self.matids = matids
        self.data = data
        self.layer = None
        self.relative_z = relative_z

    def __repr__(self):
        return "Layer<{}, {}, {}>".format(self.sl_type, self.nlayers, self.data)


def TranslatedSublayering(
    depth: float, nsublayers: int, matids: list = None, relative_z: bool = True
) -> Layer:
    """
    Creates a layer where it is directly translated in the -Z direction.
    In other words, the topography of the top layer is preserved among all
    sublayers generated.

    # Arguments
    depth: Either the minimum Z value of the new layer (relative_z = False)
    or the amount in the -Z direction the layer is translated (relative_z = True).

    nsublayers: Divides the layer into `nsublayers` equal sublayers.

    matids: list of integers equal in length to `nsublayers` with material ID
    value for that sublayer.

    relative_z: Determines how the `depth` keyword is interpreted.

    # Returns
    Layer
    """
    return Layer(
        LayerType.TRANSLATED,
        depth,
        nsublayers,
        data=[1] * nsublayers,
        matids=matids,
        relative_z=not relative_z,
    )


def ProportionalSublayering(
    depth: float, sub_thick: list, matids: list = None, relative_z: bool = True
) -> Layer:
    """
    Creates a layer where the bottom is flat, and sublayers in between are a
    depth proportional to its value in the `sub_thick` list. The Z values of
    each sublayer are an interpolation between the sublayer above and the
    bottom sublayer.

    For example, if `depth = 10`, `sub_thick = [1, 1, 0.5, 0.25]`, and
    `relative_z = True`, then four sublayers are created with depths:

    - 3.63 = 10 * 1/(1+1+0.5+0.25)
    - 3.63 = 10 * 1/(1+1+0.5+0.25)
    - 1.81 = 10 * 0.5/(1+1+0.5+0.25)
    - 0.91 = 10 * 0.25/(1+1+0.5+0.25)


    # Arguments
    depth: Either the minimum Z value of the new layer (relative_z = False)
    or the amount in the -Z direction the layer is translated (relative_z = True).

    sub_thick: Relative coefficients of depth for each sublayer. `len(sub_thick)`
    sublayers are created.

    matids: list of integers equal in length to `len(sub_thick)` with material ID
    value for that sublayer.

    relative_z: Determines how the `depth` keyword is interpreted.

    # Returns
    Layer
    """
    return Layer(
        LayerType.PROPORTIONAL,
        depth,
        len(sub_thick),
        data=sub_thick[::-1],
        matids=matids,
        relative_z=relative_z,
    )


def UniformSublayering(
    depth: float, nsublayers: int, matids: list = None, relative_z: bool = True
) -> Layer:
    """
    Creates a layer where the bottom is flat, and `nsublayers` sublayers
    are created in between. The Z values of each sublayer are an interpolation
    between the sublayer above and the bottom sublayer.

    The command:

        UniformSublayering(10., 4)

    is exactly equal to:

        ProportionalSublayering(10., [1, 1, 1, 1])

    # Arguments
    depth: Either the minimum Z value of the new layer (relative_z = False)
    or the amount in the -Z direction the layer is translated (relative_z = True).

    nsublayers: Divides the layer into `nsublayers` equal sublayers.

    matids: list of integers equal in length to `len(sub_thick)` with material ID
    value for that sublayer.

    relative_z: Determines how the `depth` keyword is interpreted.

    # Returns
    Layer
    """
    return Layer(
        LayerType.UNIFORM,
        depth,
        nsublayers,
        data=[1] * nsublayers,
        matids=matids,
        relative_z=relative_z,
    )


def stack_layers(surfmesh: Mesh, layers: List[Layer], matids: list = None) -> Mesh:
    """
    Extrudes and layers a surface mesh into a volumetric mesh, given
    the contraints in the `layers` object.
    """

    if surfmesh.element_type != ElementType.TRIANGLE:
        raise ValueError(
            "Mesh must be triangular; is actually: {}".format(surfmesh.element_type)
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

        if layer.sl_type in [LayerType.UNIFORM, LayerType.PROPORTIONAL]:
            bottom_layer.nodes[:, 2] = np.full((bottom_layer.n_nodes,), z_abs)
        elif layer.sl_type in [LayerType.TRANSLATED]:
            bottom_layer.nodes[:, 2] = bottom_layer.nodes[:, 2] - z_abs
        else:
            raise ValueError("An unknown or unsupported layer type was assigned")

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
    # vol_mesh.add_attribute(
    #    "material_id",
    #    np.repeat(np.array(mat_ids[::-1], dtype=int), elems_per_layer),
    #    type="cell",
    # )

    vol_mesh.material_id = np.repeat(
        np.array(mat_ids[::-1], dtype=int), elems_per_layer
    )

    # Each element in a layer will get its own integer value
    # This can then be used to 'mask' cells belonging to different layers
    vol_mesh._cell_layer_ids = np.repeat(
        np.array(list(range(total_layers)), dtype=int), elems_per_layer
    )

    return vol_mesh


def extrude_surface(
    surfmesh: Mesh,
    layers: list,
    matids: list = None,
    layer_type=TranslatedSublayering,
) -> Mesh:
    """
    Simple multilayer extrusion. `layers` can either be a list of depths, or
    a list of [depth, n_sublayers].

    `matids` is a list of integers with material ID values. Must be length equal to
    `len(layers)` or the sum of all `n_sublayers`.
    """

    layer_objs = []

    for (i, layer) in enumerate(layers):
        # Allow for a naive version of layering: just pass in the layer depths
        if isinstance(layer, (int, float)):
            depth = layer
            subdivisions = 1
        elif isinstance(layer, list):
            depth, subdivisions = layer

        if matids is None:
            matid = [i + 1] * subdivisions
        else:
            matid = matids[i]

        layer_objs.append(
            layer_type(depth, subdivisions, matids=[matid], relative_z=True)
        )

    return stack_layers(surfmesh, layer_objs)
