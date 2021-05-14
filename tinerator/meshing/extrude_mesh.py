from copy import deepcopy
import numpy as np
from typing import List, Union, Callable, Any
from .mesh import Mesh, StackedMesh, ElementType
from ..gis import map_elevation

LAYERING_FUNCS = {
    "constant": None,
    "snapped": None,
    "function": None,
    "raster": None,
}


def create_constant_layer(parent_layer, thickness):
    base_nodes = deepcopy(parent_layer.nodes)
    base_nodes[:, 2] -= thickness

    return base_nodes


def create_snapped_layer(parent_layer, depth):
    base_nodes = deepcopy(parent_layer.nodes)
    base_nodes[:, 2] = depth

    return base_nodes


def create_fnc_layer(parent_layer, fnc):
    base_nodes = deepcopy(parent_layer.nodes)
    x_vec = deepcopy(parent_layer.nodes[:, 0])
    y_vec = deepcopy(parent_layer.nodes[:, 1])
    base_nodes[:, 2] = fnc(x_vec, y_vec)

    return base_nodes


def create_raster_layer(parent_layer, raster):
    base_nodes = deepcopy(parent_layer.nodes)
    base_nodes[:, 2] = map_elevation(raster, base_nodes)

    return base_nodes


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


def compute_material_id(mat_ids: list, layering_stride: list, cells_per_layer: int):
    pass


def stack_layers(top_surface: Mesh, layer_nodes: list, sublayers: list, mat_ids: list):
    """
    Internal function used for stacking 2D surface meshes
    into 3D volumetric meshes.
    """

    if top_surface.element_type == ElementType.TRIANGLE:
        volume_mesh_type = ElementType.PRISM
    elif top_surface.element_type == ElementType.QUAD:
        volume_mesh_type = ElementType.HEX
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
        top_surface.elements + i * num_nodes_per_layer for i in range(num_total_layers)
    ]

    volume_nodes = np.vstack(volume_nodes)
    volume_connectivity = np.vstack(volume_connectivity)

    # Take the stacked 2D elements and connect them to make them 3D
    if volume_mesh_type == ElementType.PRISM:
        # do i need to swap connectivity? do so here
        num_prisms = num_total_layers * num_cells_per_layer
        prisms = np.zeros((num_prisms, 6), dtype=int)

        for i in range(num_prisms):
            prisms[i] = np.hstack(
                (volume_connectivity[i + num_cells_per_layer], volume_connectivity[i])
            )

        volume_connectivity = prisms
    else:
        raise NotImplementedError(f"Only triangle->prism stacking is supported")

    # Construct the final mesh object
    volume_mesh = StackedMesh(etype=volume_mesh_type)
    volume_mesh.nodes = volume_nodes
    volume_mesh.elements = volume_connectivity
    volume_mesh.material_id = compute_material_id(
        mat_ids, layer_stride, num_cells_per_layer
    )

    return volume_mesh


def extrude_mesh(
    surface_mesh: Mesh,
    layer_types: List[str],
    layer_data: List[Union[float, int, Callable]],
    sublayers: List[Union[int, float]],
    mat_ids=None,
):
    """
    Regularly extrude a 2D mesh to make a 3D mesh.

    Args
    ----

    Returns
    -------

    Examples
    --------
    """

    if not isinstance(layer_types, list):
        layer_types = [layer_types]

    if not isinstance(layer_data, list):
        layer_data = [layer_data]

    if mat_ids is None:
        mat_ids = [None] * len(layer_types)

    if not isinstance(mat_ids, list):
        mat_ids = [mat_ids]

    assert len(layer_types) == len(layer_data)
    assert len(layer_types) == len(mat_ids)

    all_layers = []
    parent_layer = Mesh(
        nodes=deepcopy(surface_mesh.nodes),
        elements=deepcopy(surface_mesh.elements),
        etype=surface_mesh.element_type,
        crs=surface_mesh.element_type,
    )

    for (l_data, l_type) in zip(layer_data, layer_types):
        try:
            layer_fnc = LAYERING_FUNCS[l_type]
        except KeyError:
            raise ValueError(f"Unsupported layer type: {l_type}")

        all_layers.append(layer_fnc(parent_layer, l_data))

        parent_layer = Mesh(
            nodes=deepcopy(all_layers[-1]),
            elements=deepcopy(surface_mesh.elements),
            etype=surface_mesh.element_type,
            crs=surface_mesh.element_type,
        )

    return stack_layers(surface_mesh, all_layers, sublayers, mat_ids)
