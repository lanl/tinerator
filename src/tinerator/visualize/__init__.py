# from .plot import plot_objects, plot_triangulation
# from .view_3d import plot_3d
# from .plot_facesets import plot_facesets
# from .plot_sets import plot_sets

import warnings
from typing import Union
from .layout_3D import get_layout as get_layout_3d

# from .config import run_server, ServerTypes, ServerSettings, set_server_settings

class MapboxStyles:
    NONE = "white-bg"
    OPEN_STREET_MAP = "open-street-map"
    CARTO_POSITRON = "carto-positron"
    CARTO_DARKMATTER = "carto-darkmatter"
    STAMEN_TERRAIN = "stamen-terrain"
    STAMEN_TONER = "stamen-toner"
    STAMEN_WATERCOLOR = "stamen-watercolor"


def mapbox_styles():
    """
    Prints available Mapbox styles. Pass one as
    an argument to ``tinerator.plot2d``,
    ``Raster.plot``, or ``Geometry.plot``.
    """
    mapbox_vars = vars(MapboxStyles)
    for key in mapbox_vars:
        if not key.startswith("__"):
            print(mapbox_vars[key])


def plot2d(
    objects: list,
    mapbox_style: str = MapboxStyles.STAMEN_TERRAIN,
    show_legend: bool = False,
    raster_cmap: list = None,
    zoom_scale=1.00,
    write_image: str = None,
    write_html: Union[str, dict] = None,
    **kwargs,
):
    """
    Plots a geospatial 2D representation of TINerator
    Geometry and Raster objects.

    ``write_image`` supports the following image formats:

    **Raster:**
    - PNG, JPEG, WebP

    **Vector:**
    - SVG, PDF
    - EPS (if Poppler library is installed)

    Colormaps should be through the Colorcet package.
    For example,

    .. code::python
        >>> import colorcet as cc
        >>> raster_cmap = cc.fire
        >>> raster_cmap = cc.swatch('fire')

    More information can be found here: "https://colorcet.holoviz.org/user_guide/index.html"

    Args:
        objects (List[tinerator.Geometry, tinerator.Raster]): [description]
        mapbox_style (str, optional): [description]. Defaults to MapboxStyles.OPEN_STREET_MAP.
        show_legend (bool, optional): [description]. Defaults to False.
        raster_cmap (list, optional): [description]. Defaults to None.
    """

    from .plotly import save, render, figure

    layout = figure(
        objects,
        mapbox_style=mapbox_style,
        show_legend=show_legend,
        raster_cmap=raster_cmap,
        zoom_scale=zoom_scale,
    )

    render(layout)

def plot3d(
    mesh,
    write_image: str = None,
    write_html: Union[str, dict] = None,
    sets: list = None,
    attribute: str = "Material Id",
    show_cube_axes: bool = False,
    show_layers_in_range: tuple = None,
    bg_color: list = None,
    **kwargs,
) -> None:
    """
    Renders a mesh in 3D using VTK.

    For ``show_layers_in_range``, it expects the following:

    .. code::python
        (layer_start, layer_stop)

    where ``layer_start`` and ``layer_stop`` are in the form:

    .. code::python
        layer_number
        layer_number.sublayer_number

    For example, to show layers between 1 and 3,

    .. code::python
        show_layers_in_range = (1, 3)

    Or to only show the first three sublayers in layer 1:

    .. code::python
        show_layers_in_range = (1.0, 1.3)

    Args:
        mesh (tinerator.Mesh): The mesh to render.
        sets (List[Union[SideSet, PointSet]], optional): Renders side sets and point sets on the mesh. Defaults to None.
        attribute (str, optional): The attribute to color the mesh by. Defaults to "Material Id".
        show_cube_axes (bool, optional): Shows cube axes around the mesh. Defaults to False.
        show_layers_in_range (tuple, optional): Only draw certain layer(s) of the mesh. Defaults to None.
    """

    from .pyvista import figure, save, display

    layout = figure(
        mesh,
        sets=sets,
        color_with_attribute=attribute,
        show_cube_axes=show_cube_axes,
        show_layers_in_range=show_layers_in_range,
        bg_color=bg_color,
    )

    if write_image:
        save(figure, **kwargs)
    else:
        display(figure)
