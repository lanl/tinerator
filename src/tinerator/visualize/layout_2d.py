import numpy as np
from collections.abc import Iterable
#import xarray
#import datashader.transfer_functions as tf
#import colorcet as cc
import plotly.graph_objects as go
#import dash_core_components as dcc
#import dash_html_components as html
from ..constants import is_tinerator_object
from ..logging import log, warn, error, debug

WGS_84 = 4326  # EPSG code
DEFAULT_RASTER_CMAP = None#cc.isolum


def get_zoom_and_center(extent, zoom_scale: float = 1.0, xp=None, fp=None):
    min_lon, min_lat, max_lon, max_lat = extent

    center = [np.mean([max_lon, min_lon]), np.mean([max_lat, min_lat])]

    area = abs(max_lon - min_lon) * abs(max_lat - min_lat)

    if xp is None:
        xp = [0, 5 ** -10, 4 ** -10, 3 ** -10, 2 ** -10, 1 ** -10, 1 ** -5]

    if fp is None:
        fp = [20, 15, 14, 13, 12, 7, 5]

    zoom = np.interp(
        x=area,
        xp=xp,
        fp=fp,
    )

    return zoom * zoom_scale, center


def add_scattermapbox(fig, xy, opacity=1.0, showlegend=True, uid=None, **kwargs):
    xy = np.array(xy)
    fig.add_trace(
        go.Scattermapbox(
            lat=xy[:, 0],
            lon=xy[:, 1],
            opacity=opacity,
            showlegend=showlegend,
            uid=uid,
            **kwargs,
        )
    )


def compute_hillshade(
    array: np.ndarray,
    azimuth: float = 315,
    angle_altitude: float = 45,
    cell_size: float = 1,
    z_scale: float = 1,
):
    azimuth = 360.0 - azimuth
    x, y = np.gradient(array * z_scale, cell_size, cell_size)
    slope = np.pi / 2.0 - np.arctan(np.sqrt(x * x + y * y))
    aspect = np.arctan2(-x, y)
    azimuthrad = azimuth * np.pi / 180.0
    altituderad = angle_altitude * np.pi / 180.0

    shaded = np.sin(altituderad) * np.sin(slope) + np.cos(altituderad) * np.cos(
        slope
    ) * np.cos((azimuthrad - np.pi / 2.0) - aspect)

    return 255 * (shaded + 1) / 2


def add_raster(
    fig,
    raster,
    name: str = None,
    colormap=None,
    uid=None,
    below="traces",
    hillshade: bool = False,
):

    if colormap is None:
        colormap = DEFAULT_RASTER_CMAP

    img = None
    min_lon, min_lat, max_lon, max_lat = raster.extent
    data = None

    if hillshade:
        data = compute_hillshade(raster.masked_data(), z_scale=5, cell_size=raster.cell_size)
    else:
        data = raster.masked_data()

    img = tf.shade(xarray.DataArray(data), cmap=colormap)
    img = img[::-1].to_pil()

    tile_layer = {
        "name": name,
        "sourcetype": "image",
        "source": img,
        "below": below,
        "coordinates": [
            # top left
            [min_lon, max_lat],
            # top right
            [max_lon, max_lat],
            # bottom right
            [max_lon, min_lat],
            # bottom left
            [min_lon, min_lat],
        ],
    }

    fig.layout["mapbox_layers"] = list(fig.layout["mapbox_layers"]) + [tile_layer]

    add_scattermapbox(
        fig,
        [[raster.centroid[0], raster.centroid[1]]],
        name=f"{name} centroid",
        opacity=0.0,
        marker_size=1,
    )


def add_geometry(fig, obj):
    gt = obj.geometry_type.strip().lower()

    if "polygon" in gt:
        coords = []
        for shp in obj.shapes:
            coords.append(np.array(shp.exterior.coords[:])[:, :2])
            coords.append(np.array([[None, None]]))

        add_scattermapbox(
            fig, np.vstack(coords), name="Polygon", mode="lines", fill="toself"
        )
    elif "line" in gt:
        coords = []

        for shp in obj.shapes:
            coords.append(np.array(shp.coords[:])[:, :2])
            coords.append(np.array([[None, None]]))

        add_scattermapbox(fig, np.vstack(coords), name="Lines", mode="lines+markers")
    elif "point" in gt:
        coords = []

        for shp in obj.shapes:
            coords.append(np.array(shp.coords[:])[:, :2])
            coords.append(np.array([[None, None]]))

        add_scattermapbox(
            fig, np.vstack(coords), name="Points", mode="markers", marker_size=10
        )
    else:
        raise ValueError(f"Unknown geometry type: {obj.geometry_type}")


def init_figure(
    objects,
    raster_cmap=None,
    mapbox_style=None,
    show_legend=False,
    margin=7,
    zoom_scale=1.0,
):
    debug(
        f"Initializing figure. {objects=}, {raster_cmap=}, "
        f"{mapbox_style=}, {show_legend=}, {zoom_scale=}"
    )

    if mapbox_style is None:
        mapbox_style = "white-bg"

    debug(f'Mapbox style: "{mapbox_style}"')

    if not isinstance(objects, Iterable):
        objects = [objects]

    geom_objs = []
    tile_objs = []
    extents = []

    for obj in objects:
        if is_tinerator_object(obj, "Geometry"):
            geom_objs.append(obj)
        elif is_tinerator_object(obj, "Raster"):
            tile_objs.append(obj)

    fig = go.Figure()
    fig.layout["mapbox_layers"] = []

    for g in geom_objs:
        g = g.reproject(WGS_84)
        extent = g.extent
        extents.append([extent[1], extent[0], extent[3], extent[2]])
        add_geometry(fig, g)

    for (i, t) in enumerate(tile_objs):
        t = t.reproject(WGS_84)
        extents.append(t.extent)
        add_raster(fig, t, colormap=raster_cmap)

    extents = np.vstack(extents)
    map_extent = [*np.min(extents[:, :2], axis=0), *np.max(extents[:, 2:], axis=0)]
    zoom, map_center = get_zoom_and_center(map_extent, zoom_scale=zoom_scale)

    fig.update_layout(
        geo={
            "fitbounds": "locations",
        },
        margin={
            "r": margin,
            "t": margin,
            "l": margin,
            "b": margin,
        },
        mapbox={
            "style": mapbox_style,
            "center": {
                "lon": map_center[0],
                "lat": map_center[1],
            },
            "zoom": zoom,
        },
        showlegend=show_legend,
    )

    return fig


def get_layout(
    objects,
    mapbox_style=None,
    show_legend=False,
    raster_cmap=None,
    width: str = "100%",
    height: str = "calc(100vh - 0px)",
    zoom_scale: float = 1.0,
    **kwargs,
):
    return html.Div(
        style={"width": width, "height": height},
        children=[
            dcc.Graph(
                figure=init_figure(
                    objects,
                    raster_cmap=raster_cmap,
                    mapbox_style=mapbox_style,
                    show_legend=show_legend,
                    zoom_scale=zoom_scale,
                    **kwargs,
                ),
                config={
                    "showAxisDragHandles": True,
                    "watermark": False,
                    "autosizable": True,
                    "displaylogo": False,
                    "fillFrame": True,
                    "responsive": True,
                    "staticPlot": False,
                },
            ),
        ],
    )
