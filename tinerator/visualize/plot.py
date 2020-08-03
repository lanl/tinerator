from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os
import json

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


def plot_raster(
    raster,
    title=None,
    xlabel=None,
    ylabel=None,
    extent=[],
    outfile=None,
    geometry=None,
):
    """
    Plots a raster matrix.
    """

    if not extent:
        extent = (0, np.shape(raster)[1], 0, np.shape(raster)[0])

    vmin, vmax = np.nanmin(raster), np.nanmax(raster)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    __apply_grid_to_axis(ax)

    cax = ax.imshow(
        raster, zorder=9, extent=extent, vmin=vmin, vmax=vmax, cmap=topocmap
    )

    cbar = fig.colorbar(cax, ax=ax)
    # Raster does not necessarily display elevation
    # cbar.set_label("Elevation (m)", rotation=270)

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

    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if outfile is not None:
        fig.savefig(outfile)
    else:
        plt.show()
