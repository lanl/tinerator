import numpy as np
from collections.abc import Iterable
import xarray
import datashader.transfer_functions as tf
import colorcet as cc
import plotly.graph_objects as go
import dash_core_components as dcc
import dash_html_components as html
from ..constants import is_tinerator_object
from ..logging import log, warn, error, debug

WGS_84 = 4326  # EPSG code
DEFAULT_RASTER_CMAP = cc.isolum


def get_zoom_and_center(extent, zoom_scale: float = 1.0, xp=None, fp=None):
    min_x, min_y, max_x, max_y = extent
    area = abs(max_x - min_x) * abs(max_y - min_y)

    center = (min_x + (max_x - min_x) / 2.0, min_y + (max_y - min_y) / 2.0)

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


def add_lines(fig, xy, uid=None):
    xy = np.array(xy)

    fig.add_trace(
        go.Scattermapbox(
            name="Lines",
            mode="lines+markers",
            lon=xy[:, 0],
            lat=xy[:, 1],
            opacity=1.0,
            showlegend=True,
            uid=uid,
        )
    )


def add_polygon(fig, xy_exterior, xy_interior=None, uid=None):
    xy_exterior = np.array(xy_exterior)
    # xy = np.array(shape.exterior.coords[:])
    # xy1 = np.array(shape.interior.coords)

    fig.add_trace(
        go.Scattermapbox(
            name="Polygon",
            mode="lines",
            fill="toself",
            lon=xy_exterior[:, 0],
            lat=xy_exterior[:, 1],
            opacity=1.0,
            showlegend=True,
            uid=uid,
        )
    )


def add_points(fig, xy, uid=None):
    xy = np.array(xy)

    fig.add_trace(
        go.Scattermapbox(
            name="Points",
            mode="markers",
            lon=xy[:, 0],
            lat=xy[:, 1],
            opacity=1.0,
            showlegend=True,
            uid=uid,
        )
    )


def add_raster(raster, name: str = None, colormap=None, uid=None, below="traces"):

    if colormap is None:
        colormap = DEFAULT_RASTER_CMAP

    raster = raster.reproject(WGS_84)
    min_x, max_y, max_x, min_y = raster.extent

    xarr = xarray.DataArray(raster.masked_data())
    img = tf.shade(xarr, cmap=colormap)[::-1].to_pil()

    return {
        "name": name,
        "sourcetype": "image",
        "source": img,
        "below": below,
        "coordinates": [
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y],
        ],
    }


def add_geometry(fig, obj):
    gt = obj.geometry_type.strip().lower()
    obj = obj.reproject(WGS_84)

    if "polygon" in gt:
        for shp in obj.shapes:
            add_polygon(
                fig,
                shp.exterior.coords[:],
                # xy_interior=shp.interior.coords[:]
            )
    elif "line" in gt:
        coords = []

        for shp in obj.shapes:
            coords.append(np.array(shp.coords[:])[:, :2])
            coords.append(np.array([[None, None]]))

        add_lines(fig, np.vstack(coords))
    elif "point" in gt:
        for shp in obj.shapes:
            add_points(fig, shp.coords[:])
    else:
        raise ValueError(f"Unknown geometry type: {obj.geometry_type}")


def init_figure(
    objects, raster_cmap=None, mapbox_style=None, show_legend=False, margin=7, zoom_scale = 1.0,
):
    fig = go.Figure()

    map_extent = [+1e8, +1e8, -1e8, -1e8]
    mapbox_layers = []

    if mapbox_style is None:
        mapbox_style = "white-bg"

    log(f'Mapbox style: "{mapbox_style}"')

    if not isinstance(objects, Iterable):
        objects = [objects]

    if not isinstance(raster_cmap, Iterable):
        raster_cmap = [raster_cmap]

    rcmap_idx = 0

    for (i, obj) in enumerate(objects):
        extent = obj.extent
        map_extent[0] = min(map_extent[0], extent[0])
        map_extent[1] = min(map_extent[1], extent[1])
        map_extent[2] = max(map_extent[2], extent[2])
        map_extent[3] = max(map_extent[3], extent[3])

        if is_tinerator_object(obj, "Geometry"):
            add_geometry(fig, obj)
        elif is_tinerator_object(obj, "Raster"):
            r_cmap = None

            try:
                r_cmap = raster_cmap[rcmap_idx]
                rcmap_idx += 1
            except IndexError as e:
                pass

            layer = add_raster(obj, colormap=r_cmap)
            mapbox_layers.append(layer)
        else:
            raise ValueError(f"Unknown object type: {type(obj)}")

    zoom, map_center = get_zoom_and_center(map_extent, zoom_scale=zoom_scale)

    #fig.update_geos(fitbounds="locations")
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
        mapbox_layers=mapbox_layers,
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
