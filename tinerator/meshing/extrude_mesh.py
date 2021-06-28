from copy import deepcopy
import numpy as np
from typing import List, Union, Callable, Any
from .mesh import Mesh, StackedMesh
from .meshing_types import ElementType
from ..gis import map_elevation, reproject_raster
from ..logging import log, warn, debug, error


def create_constant_layer(parent_layer, thickness, **kwargs):
    base_nodes = deepcopy(parent_layer.nodes)
    base_nodes[:, 2] -= thickness

    return base_nodes


def create_snapped_layer(parent_layer, depth, **kwargs):
    base_nodes = deepcopy(parent_layer.nodes)
    base_nodes[:, 2] = depth

    return base_nodes


def create_fnc_layer(parent_layer, fnc, **kwargs):
    base_nodes = deepcopy(parent_layer.nodes)
    x_vec = deepcopy(parent_layer.nodes[:, 0])
    y_vec = deepcopy(parent_layer.nodes[:, 1])
    base_nodes[:, 2] = fnc(x_vec, y_vec)

    return base_nodes


def create_raster_layer(
    parent_layer, raster, dest_crs=None, raster_thickness: bool = True, **kwargs
):

    if dest_crs is not None:
        raster = reproject_raster(raster, dest_crs)

    base_nodes = deepcopy(parent_layer.nodes)

    if raster_thickness:
        base_nodes[:, 2] -= map_elevation(raster, base_nodes)
    else:
        base_nodes[:, 2] = map_elevation(raster, base_nodes)

    return base_nodes


def create_raster_thickness_layer(parent_layer, raster, **kwargs):
    return create_raster_layer(parent_layer, raster, raster_thickness=True, **kwargs)


def create_raster_depth_layer(parent_layer, raster, **kwargs):
    return create_raster_layer(parent_layer, raster, raster_thickness=False, **kwargs)


def interpolate_sublayers(top_surface: Mesh, layer_nodes: list, sublayers: list):
    """
    Internal function used for computing
    the sublayering: i.e., the intermediate layers
    between two major layers.
    """
    major_layers = [deepcopy(top_surface.nodes)] + layer_nodes
    all_layers = []
    layer_stride = []

    for i in range(len(major_layers) - 1):
        local_top = major_layers[i]
        local_bottom = major_layers[i + 1]

        local_layering = sublayers[i]

        if isinstance(local_layering, int):
            local_layering -= 1

        if local_layering is None or local_layering == 0:
            middle_layers = []
        else:
            if isinstance(local_layering, int):
                local_layering = [
                    (i + 1) / (local_layering + 1) for i in range(local_layering)
                ]

            local_layering = sorted(local_layering)

            assert all(
                [x < 1.0 and x > 0.0 for x in local_layering]
            ), "local_layers must be an int or between [0, 1]"

            middle_layers = [
                local_bottom + (local_top - local_bottom) * x for x in local_layering
            ][::-1]

        all_layers.extend([local_top, *middle_layers])
        layer_stride.append(len(middle_layers) + 1)

    all_layers.append(major_layers[-1])

    return all_layers, layer_stride


def compute_layer_id(layering_stride: list, cells_per_layer: int, nodes_per_layer: int):
    """
    Computes the layering schema for cells and nodes.
    """
    layer_schema = []
    for (i, stride) in enumerate(layering_stride):
        for j in range(stride):
            layer_schema.append(float(f"{i+1}.{j+1}"))

    bottom_node_layer = len(layering_stride) + 1.0

    cell_layer_id = np.repeat(layer_schema, cells_per_layer).astype(float)
    node_layer_id = np.repeat(
        layer_schema + [bottom_node_layer], nodes_per_layer
    ).astype(float)
    return cell_layer_id, node_layer_id


def compute_material_id(mat_ids: list, layering_stride: list, cells_per_layer: int):
    """
    Computes the material ID for generated sublayers.
    """

    mat_ids_all = []

    for i in range(len(layering_stride)):
        mat_id = mat_ids[i]
        stride = layering_stride[i]

        if mat_id is None:
            mat_id = [i + 1] * stride
        elif isinstance(mat_id, (int, np.int)):
            mat_id = [mat_id] * stride

        assert len(mat_id) == stride
        mat_ids_all.extend(mat_id)

    return np.repeat(mat_ids_all, cells_per_layer).astype(int)


def stack_layers(
    top_surface: Mesh, layer_nodes: list, sublayers: list, mat_ids: list, crs=None
):
    """
    Internal function used for stacking 2D surface meshes
    into 3D volumetric meshes.
    """

    if top_surface.element_type == ElementType.TRIANGLE:
        volume_mesh_type = ElementType.PRISM
        volume_mesh_nodes_per_elem = 6
    elif top_surface.element_type == ElementType.QUAD:
        volume_mesh_type = ElementType.HEX
        volume_mesh_nodes_per_elem = 8
    else:
        raise ValueError(
            f"Could not create stacked mesh from {top_surface.element_type}"
        )

    assert len(sublayers) == len(layer_nodes)
    assert len(mat_ids) == len(sublayers)

    # Use the sublayering list to compute the intermediate layers
    volume_nodes, layer_stride = interpolate_sublayers(
        top_surface, layer_nodes, sublayers
    )

    num_nodes_per_layer = top_surface.n_nodes
    num_cells_per_layer = top_surface.n_elements
    num_total_layers = np.sum(layer_stride)

    # Translate local layer element connectivity and make it global
    volume_connectivity = [
        top_surface.elements + i * num_nodes_per_layer
        for i in range(num_total_layers + 1)
    ]

    volume_nodes = np.vstack(volume_nodes)
    volume_connectivity = np.vstack(volume_connectivity)

    # Take the stacked 2D elements and connect them to make them 3D
    if volume_mesh_type in [ElementType.PRISM, ElementType.HEX]:
        num_prisms = num_total_layers * num_cells_per_layer
        prisms = np.zeros((num_prisms, volume_mesh_nodes_per_elem), dtype=int)

        for i in range(num_prisms):
            prisms[i] = np.hstack(
                (volume_connectivity[i + num_cells_per_layer], volume_connectivity[i])
            )

        volume_connectivity = prisms
    else:
        raise NotImplementedError(f"Only triangle->prism stacking is supported")

    # Construct the final mesh object
    volume_mesh = StackedMesh(etype=volume_mesh_type)
    volume_mesh.crs = crs
    volume_mesh.nodes = volume_nodes
    volume_mesh.elements = volume_connectivity
    volume_mesh.material_id = compute_material_id(
        mat_ids, layer_stride, num_cells_per_layer
    )

    cell_layer_id, node_layer_id = compute_layer_id(
        layer_stride, num_cells_per_layer, num_nodes_per_layer
    )

    layertyp = np.zeros((volume_mesh.n_nodes,), dtype=int)
    layertyp[:num_nodes_per_layer] = -2
    layertyp[-num_nodes_per_layer:] = -1

    volume_mesh.add_attribute("layertyp", layertyp, type="node")
    volume_mesh.add_attribute(
        "cell_layer_id", cell_layer_id, type="cell", data_type=float
    )
    volume_mesh.add_attribute(
        "node_layer_id", node_layer_id, type="node", data_type=float
    )

    return volume_mesh


LAYERING_FUNCS = {
    "constant": create_constant_layer,
    "snapped": create_snapped_layer,
    "function": create_fnc_layer,
    "raster": create_raster_layer,
    "raster-constant": create_raster_thickness_layer,
    "raster-snapped": create_raster_thickness_layer,
}


def extrude_mesh(
    surface_mesh: Mesh,
    layers: List[List[Any]],
):
    """
    Regularly extrude a 2D mesh to make a 3D mesh.

    ``layers`` should be a list containing tuples of attributes
    for each layer to add, in the form:

        (layer_type, layer_data, sublayers, mat_ids)

    where ``layer_type`` is one of:

        - "constant": a layer translated in -Z
        - "snapped": a layer where all nodes are set to Z
        - "function": a layer where the Z value of nodes are set from ``f(x, y) -> z``
        - "raster": a layer where the Z values of nodes are **subtracted** from the closest raster cell
        - "raster-constant": the same as "raster"
        - "raster-snapped": a layer where the Z values of nodes are set from raster cells

    +-------------------+--------------------------------------------------+
    | ``layer_type``    | ``layer_data``                                   |
    +===================+==================================================+
    | "constant"        | depth to extrude                                 |
    +-------------------+--------------------------------------------------+
    | "snapped"         | value to set layer node Z values to              |
    +-------------------+--------------------------------------------------+
    | "function"        | a Python function of the form ``f(x,y) -> z``    |
    +-------------------+--------------------------------------------------+
    | "raster-constant" | a TINerator Raster object                        |
    +-------------------+--------------------------------------------------+
    | "raster-snapped"  | a TINerator Raster object                        |
    +-------------------+--------------------------------------------------+

    Args
    ----
        surface_mesh (tinerator.meshing.Mesh): A triangular surface mesh to extrude.
        layers (List[List[Any]]): A list of layer schemas, in the
            form ``(type, data, sublayers, material_id)``

    Returns
    -------
        tinerator.meshing.Mesh: A prism mesh formed from stacked layers

    Examples
    --------
        >>> fnc = lambda x, y: x**2. + y**2.
        >>> subsurface = tin.gis.load_raster("subsurface.tif")
        >>> layers = [
                ("constant", 50., 2, 1),
                ("snapped", 12.5, [0.25, 0.25, 0.5], 3),
                ("function", fnc, 2, [4,5]),
                ("raster", subsurface, 1, 6),
            ]
        >>> vol_mesh = tin.meshing.extrude_mesh(triangulation, layers)
    """

    # TODO: how do you handle negative or zero volume prisms?
    # TODO: how do you handle pinchouts?

    if not isinstance(layers, (tuple, list)):
        raise ValueError(f"Cannot parse layers. View function help.")

    if not isinstance(layers[0], (list, tuple)):
        layers = [layers]

    layer_types = []
    layer_data = []
    layer_sublayers = []
    layer_material_ids = []

    for layer in layers:
        assert (
            len(layer) == 4
        ), "layer must be (layer_type, layer_data, sublayers, material_ids)"
        layer_types.append(layer[0])
        layer_data.append(layer[1])
        layer_sublayers.append(layer[2])
        layer_material_ids.append(layer[3])

    all_layers = []
    parent_layer = Mesh(
        nodes=deepcopy(surface_mesh.nodes),
        elements=deepcopy(surface_mesh.elements),
        etype=surface_mesh.element_type,
        crs=surface_mesh.crs,
    )

    for (l_data, l_type) in zip(layer_data, layer_types):
        debug(f'Generating layer "{l_type}" with data {l_data}')

        try:
            layer_fnc = LAYERING_FUNCS[l_type]
        except KeyError:
            raise ValueError(f"Unsupported layer type: {l_type}")

        all_layers.append(layer_fnc(parent_layer, l_data, dest_crs=surface_mesh.crs))

        parent_layer = Mesh(
            nodes=deepcopy(all_layers[-1]),
            elements=deepcopy(surface_mesh.elements),
            etype=surface_mesh.element_type,
            crs=surface_mesh.element_type,
        )

    return stack_layers(
        surface_mesh,
        all_layers,
        layer_sublayers,
        layer_material_ids,
        crs=surface_mesh.crs,
    )
