from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LightSource
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import json
from ..gis import Raster, Geometry, parse_crs, reproject_geometry, reproject_raster
from typing import Union

with open(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "schwarzwald_topographie.json",
    ),
    "r",
) as f:
    cmap_data = json.loads(f.read())

topocmap = mcolors.LinearSegmentedColormap(
    name="schwarzwald_topographie", segmentdata=cmap_data
)


def plot_triangulation(
    nodes: np.array,
    triangles: np.array,
    face_attribute: np.array = None,
    title: str = None,
    histogram_bins: int = None,
    histogram_range: tuple = None,
):
    """
    Plots a triangulation using Matplotlib.
    Optionally, can plot face color using some attribute along
    with a histogram plot of that face attribute.
    """
    x = nodes[:, 0]
    y = nodes[:, 1]

    fig, ax_top = plt.subplots(figsize=(8, 10))

    ax_top.set_aspect("equal")
    tc = ax_top.tripcolor(x, y, triangles, facecolors=face_attribute, edgecolors="k")

    if title is not None:
        ax_top.set_title(title)

    if face_attribute is not None and histogram_bins is not None:
        divider = make_axes_locatable(ax_top)
        ax_bot = divider.append_axes("bottom", size=0.8, pad=0.3)
        cax = divider.append_axes("right", size=0.08, pad=0.1)

        ax_bot.hist(
            face_attribute,
            bins=histogram_bins,
            range=histogram_range,
            histtype="bar",
            orientation="vertical",
            ec="w",
            fc="gray",
        )
        cbar = fig.colorbar(tc, cax=cax)
    else:
        cbar = fig.colorbar(tc, cax=ax_top)

    plt.show()


def __apply_grid_to_axis(axis):
    axis.set_facecolor("#EAEAF1")
    axis.grid("on", zorder=0, color="white")


def __init_figure(
    title=None, xlabel=None, ylabel=None, figsize=(12, 8), apply_grid=True
):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    if apply_grid:
        __apply_grid_to_axis(ax)

    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    return fig, ax

# https://matplotlib.org/2.0.2/examples/specialty_plots/topographic_hillshading.html
def __add_raster_obj(
    fig,
    ax,
    raster,
    hillshade=False,
    cell_size=(1, 1),
    extent=(),
    zorder: int = 9,
):

    if not extent:
        extent = (0, np.shape(raster)[1], 0, np.shape(raster)[0])

    vmin, vmax = np.nanmin(raster), np.nanmax(raster)

    if hillshade:
        dx, dy = cell_size
        vertical_exaggeration = 1.0

        ls = LightSource(azdeg=315, altdeg=45)

        hillshade_raster = ls.hillshade(
            raster, vert_exag=vertical_exaggeration, dx=dx, dy=dy
        )

        cax = ax.imshow(
            hillshade_raster * vmax + vmin,
            cmap="gray",
            zorder=zorder,
            extent=extent,
        )
    else:
        cax = ax.imshow(
            raster,
            zorder=zorder,
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            cmap=topocmap,
        )

    cbar = fig.colorbar(cax, ax=ax)
    # Raster does not necessarily display elevation
    # cbar.set_label("Elevation (m)", rotation=270)


def __add_vector_obj(fig, ax, shape, zorder=10):

    shape_type = shape.geometry_type

    if "Point" in shape_type:
        raise NotImplementedError
    elif "LineString" in shape_type:
        for shp in shape.shapes:
            coords = np.array(shp.coords[:])
            ax.plot(coords[:, 0], coords[:, 1], zorder=zorder, marker="o")
    elif "Polygon" in shape_type:
        for shp in shape.shapes:
            xs, ys = shp.exterior.xy
            ax.fill(xs, ys, alpha=0.5, zorder=zorder, linewidth=1.2, fc="r", ec="black")
    elif "GeometryCollection" in shape_type:
        raise NotImplementedError
    else:
        raise ValueError(f"Unknown shape type: {shape_type}")


def plot_objects(
    objects: list,
    zorder: list = None,
    outfile: str = None,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    raster_hillshade: bool = False,
    crs: Union[str, int, dict] = None,
):

    if crs is None:
        crs = objects[0].crs
    else:
        crs = parse_crs(crs)

    if title is None:
        title = f"CRS: {crs.name}"

    fig, ax = __init_figure(title=title, xlabel=xlabel, ylabel=ylabel)

    if zorder is not None:
        assert len(zorder) == len(objects), "`zorder` and `objects` differ in length"

    for obj in objects:
        if isinstance(obj, Geometry):
            obj = reproject_geometry(obj, crs)
            __add_vector_obj(fig, ax, obj)
        elif isinstance(obj, Raster):
            obj = reproject_raster(obj, crs)
            extent = obj.extent
            extent = [extent[0], extent[2], extent[1], extent[3]]
            cs = (obj.cell_size, obj.cell_size)
            __add_raster_obj(
                fig,
                ax,
                obj.masked_data(),
                hillshade=raster_hillshade,
                cell_size=cs,
                extent=extent,
            )
        else:
            print("WARNING: non-plottable object passed.")

    if outfile is not None:
        fig.savefig(outfile)
    else:
        plt.show()
