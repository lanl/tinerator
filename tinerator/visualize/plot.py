from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LightSource
import numpy as np
import os
import json
from ..gis import Raster, Shape

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


def __apply_grid_to_axis(axis):
    axis.set_facecolor("#EAEAF1")
    axis.grid("on", zorder=0, color="white")

def __init_figure(title=None, xlabel=None, ylabel=None, figsize=(12, 8), apply_grid=True):
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


def __add_raster_obj(fig, ax, raster, hillshade=False, cell_size=(1,1), extent=()):

    if not extent:
        extent = (0, np.shape(raster)[1], 0, np.shape(raster)[0])

    vmin, vmax = np.nanmin(raster), np.nanmax(raster)

    if hillshade:
        dx, dy = cell_size
        vertical_exaggeration = 1.

        ls = LightSource(azdeg=315, altdeg=45)

        hillshade_raster = ls.hillshade(
            raster, 
            vert_exag=vertical_exaggeration, 
            dx=dx, 
            dy=dy
        )

        cax = ax.imshow(
            hillshade_raster * vmax + vmin, 
            cmap='gray', 
            zorder=9, 
            extent=extent
        )
    else:
        cax = ax.imshow(
            raster, zorder=9, extent=extent, vmin=vmin, vmax=vmax, cmap=topocmap
        )

    cbar = fig.colorbar(cax, ax=ax)
    # Raster does not necessarily display elevation
    # cbar.set_label("Elevation (m)", rotation=270)

def __add_vector_obj(fig, ax, points: np.ndarray):
    ax.scatter(points[:,0], points[:,1], zorder=9, s=3.0, c="red")

    '''
        if geometry is not None:
        for g in geometry:
            gg = g["coordinates"]
            geom_type = g["type"].lower().strip()
            if geom_type == "polygon":
                ax.fill(
                    gg[:, 0],
                    gg[:, 1],
                    zorder=99,
                    edgecolor="black",
                    linewidth=1.2,
                )
            elif geom_type == "linestring" or geom_type == "line":
                ax.plot(gg[:, 0], gg[:, 1], zorder=99, marker="o")
            else:
                print("Unknown geometry type")
                ax.scatter(gg[:, 0], gg[:, 1], zorder=99)
    '''

def plot_objects(
        objects: list, 
        zorder: list = None,
        outfile: str = None,
        title: str = None,
        xlabel:str=None,
        ylabel:str=None,
        extent:tuple=(),
        raster_hillshade:bool=False,
        raster_cellsize:tuple=(1, 1)
    ):

    fig, ax = __init_figure(title=title, xlabel=xlabel, ylabel=ylabel)

    if zorder is not None:
        assert len(zorder) == len(objects), '`zorder` and `objects` differ in length'

    for obj in objects:
        if isinstance(obj, Shape):
            __add_vector_obj(fig, ax, obj.points)
        elif isinstance(obj, Raster):
            __add_raster_obj(fig, ax, obj.masked_data(), hillshade=raster_hillshade, cell_size=raster_cellsize, extent=extent)
        else:
            print('WARNING: non-plottable object passed.')

    if outfile is not None:
        fig.savefig(outfile)
    else:
        plt.show()