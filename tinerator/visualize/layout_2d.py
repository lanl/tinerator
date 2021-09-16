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
            lat=xy[:, 0],
            lon=xy[:, 1],
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
            lat=xy_exterior[:, 0],
            lon=xy_exterior[:, 1],
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
            marker_size=10,
            lat=xy[:, 0],
            lon=xy[:, 1],
            opacity=1.0,
            showlegend=True,
            uid=uid,
        )
    )


def add_raster(
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
    min_lat, max_lon, max_lat, min_lon = raster.extent

    if hillshade:
        from matplotlib.colors import LightSource
        import matplotlib.pyplot as plt
        from PIL import Image

        data = raster.masked_data()
        dx = dy = raster.cell_size
        dy = 111200 * dy
        dx = 111200 * dx * np.cos(np.radians(min_lon))
        ls = LightSource(azdeg=315, altdeg=45)

        cmap = plt.cm.gist_earth

        rgb = ls.shade(
            data,
            cmap=cmap,
            vmin=np.nanmin(data),
            vmax=np.nanmax(data),
            blend_mode="overlay",
            vert_exag=5,
            dx=dx,
            dy=dy,
        )
        rgb[:, :, :3] *= 255
        img = Image.fromarray(rgb[::-1], mode="RGBA")
        img.show()
    else:
        xarr = xarray.DataArray(raster.masked_data())
        img = tf.shade(xarr, cmap=colormap)
        img = img[::-1].to_pil()

    return {
        "name": name,
        "sourcetype": "image",
        "source": img,
        "below": below,
        "coordinates": [
            [min_lat, min_lon],
            [max_lat, min_lon],
            [max_lat, max_lon],
            [min_lat, max_lon],
        ],
    }


def add_geometry(fig, obj):
    gt = obj.geometry_type.strip().lower()

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
        coords = []

        for shp in obj.shapes:
            coords.append(np.array(shp.coords[:])[:, :2])
            coords.append(np.array([[None, None]]))

        add_points(fig, np.vstack(coords))
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
    fig = go.Figure()

    map_extent = [+1e8, +1e8, -1e8, -1e8]
    mapbox_layers = []

    if mapbox_style is None:
        mapbox_style = "white-bg"

    debug(f'Mapbox style: "{mapbox_style}"')

    if not isinstance(objects, Iterable):
        objects = [objects]

    if not isinstance(raster_cmap, Iterable):
        raster_cmap = [raster_cmap]

    rcmap_idx = 0

    for (i, obj) in enumerate(objects):
        obj = obj.reproject(WGS_84)

        extent = list(obj.extent)
        debug(f"{extent=}")

        if is_tinerator_object(obj, "Geometry"):
            add_geometry(fig, obj)
        elif is_tinerator_object(obj, "Raster"):
            # ============================ #
            # TODO: Raster extent needs to be flipped
            # Band-aid
            min_lon, min_lat, max_lon, max_lat = extent
            extent = [min_lat, min_lon, max_lat, max_lon]
            # ============================ #

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

        map_extent[0] = min(map_extent[0], extent[0])
        map_extent[1] = min(map_extent[1], extent[1])
        map_extent[2] = max(map_extent[2], extent[2])
        map_extent[3] = max(map_extent[3], extent[3])

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
                "lat": map_center[0],
                "lon": map_center[1],
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
