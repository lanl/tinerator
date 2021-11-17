import pyvista as pv
import numpy as np
from collections.abc import Iterable


class MeshViewType:
    POINTS = 0
    WIREFRAME = 1
    SURFACE = 2


class ScalarMode:
    DEFAULT = 0
    USE_POINT_DATA = 1
    USE_CELL_DATA = 2
    USE_POINT_FIELD_DATA = 3
    USE_CELL_FIELD_DATA = 4
    USE_FIELD_DATA = 5


class GetArray:
    BY_ID = 0
    BY_NAME = 1


def prepare_vtk_mesh(
    plotter: pv.Plotter,
    mesh,
    representation=MeshViewType.SURFACE,
    opacity=1.0,
    color_attribute=None,
    showCubeAxes=False,
    colorMapPreset="gist_earth",
    color_range=None,
    custom_fields: dict = None,
    threshold: dict = None,
):
    """

    threshold must be in the form:
    """
    vtk_mesh = mesh.to_vtk_mesh()

    if threshold is not None:
        vtk_mesh = vtk_mesh.threshold(
            value=threshold["value"],
            scalars=threshold["scalars"],
            invert=threshold["invert"],
            continuous=threshold["continuous"],
        )

    if custom_fields is not None:
        for key in custom_fields:
            vtk_mesh[key] = custom_fields[key]

    mesh_state = None

    if color_attribute is None:
        active_scalar = None
        # mesh_state = to_mesh_state(vtk_mesh)
        color_range = [0, 1]
    else:
        active_scalar = color_attribute
        # mesh_state = to_mesh_state(vtk_mesh, field_to_keep=color_attribute)

        if color_range is None:
            try:
                color_range = [
                    np.nanmin(vtk_mesh[color_attribute]),
                    np.nanmax(vtk_mesh[color_attribute]),
                ]
            except KeyError:
                pass

    plotter.add_mesh(
        vtk_mesh,
        opacity=opacity,
        show_edges=True,
        cmap=colorMapPreset,
        scalars=active_scalar,
    )

    # return dash_vtk.GeometryRepresentation(
    #    showCubeAxes=showCubeAxes,
    #    colorMapPreset=colorMapPreset,
    #    colorDataRange=color_range,
    #    children=[dash_vtk.Mesh(state=mesh_state)],
    #    property={
    #        "edgeVisibility": True,
    #        "opacity": opacity,
    #        "representation": representation,
    #    },
    # )


def vtk_view(
    mesh,
    color_attribute=None,
    sets=None,
    show_cube_axes=False,
    show_layers_in_range=None,
    bg_color: list = None,
    window_size: tuple = None,
):
    plotter = pv.Plotter(window_size=window_size)
    plotter.background_color = bg_color
    plotter.enable_anti_aliasing()

    threshold = None
    primary_opacity = 1.0
    primary_color_attribute = color_attribute
    primary_representation = MeshViewType.SURFACE

    if bg_color is None:
        bg_color = [0.2, 0.3, 0.4]

    if sets or show_layers_in_range:
        primary_opacity = 0.25
        primary_representation = MeshViewType.WIREFRAME
        primary_color_attribute = None

    prepare_vtk_mesh(
        plotter,
        mesh,
        opacity=primary_opacity,
        representation=primary_representation,
        color_attribute=primary_color_attribute,
        showCubeAxes=show_cube_axes,
    )

    if show_layers_in_range:
        prepare_vtk_mesh(
            plotter,
            mesh,
            opacity=1.0,
            representation=MeshViewType.SURFACE,
            color_attribute=color_attribute,
            threshold={
                "value": show_layers_in_range,
                "scalars": "cell_layer_id",
                "invert": True,
                "continuous": True,
            },
        )

    if isinstance(sets, Iterable):
        for (set_id, set_mesh) in enumerate(sets):
            num_cells = len(set_mesh.primary_cells)

            prepare_vtk_mesh(
                plotter,
                set_mesh,
                color_attribute="set_id",
                custom_fields={"set_id": np.array([set_id + 1] * num_cells, dtype=int)},
                color_range=[1, len(sets)],
                opacity=1.0,
                representation=MeshViewType.SURFACE,
                showCubeAxes=False,
            )

    # cameraParallelProjection (boolean; default False): Use parallel projection (default: False).
    # cameraPosition (list; default \[0, 0, 1\]): Initial camera position from an object in [0,0,0].
    # cameraViewUp (list; default \[0, 1, 0\]): Initial camera position from an object in [0,0,0].

    # savefig: https://discourse.vtk.org/t/save-window-rendering-results-to-image/3772/2
    # or: https://kitware.github.io/vtk-js/api/Common_Core_ImageHelper.html
    # or: https://github.com/Kitware/vtk-js/issues/1598

    return plotter


def get_layout(
    mesh,
    sets=None,
    color_with_attribute="Material Id",
    show_cube_axes=False,
    show_layers_in_range: tuple = None,
    bg_color: list = None,
    window_size: tuple = (600, 400),
):

    if isinstance(color_with_attribute, str):
        mat = (
            color_with_attribute.lower()
            .strip()
            .replace(" ", "")
            .replace("_", "")
            .replace("-", "")
        )

        if mat == "materialid":
            color_with_attribute = "Material Id"

    return vtk_view(
        mesh,
        sets=sets,
        color_attribute=color_with_attribute,
        show_cube_axes=show_cube_axes,
        show_layers_in_range=show_layers_in_range,
        window_size=window_size,
    )
