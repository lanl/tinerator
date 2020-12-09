import numpy as np
from matplotlib import pyplot as plt


def plot_facesets(dem_object, fs_list):
    """
    Displays a topological preview of how facesets will look after
    Exodus generation.

    # Arguments
    dem_object (tinerator.DEM): A DEM class instance
    fs_list (list<tinerator.Faceset>): One or more initialized Faceset objects
    """
    import warnings

    np.warnings.filterwarnings("ignore")

    cmap = "tab10"

    def plot_row(fs_object, row, extent=None):
        empty = np.zeros(np.shape(dem))
        empty.fill(np.nan)

        if fs_object._has_type == "__NAIVE":

            if fs_object._metadata["top"]:
                row[0].imshow(dem, cmap=cmap, extent=extent)
            else:
                row[0].imshow(empty, extent=extent)

            if fs_object._metadata["sides"]:
                row[1].scatter(
                    dem_object.boundary[:, 0], dem_object.boundary[:, 1]
                )
                row[1].set_aspect(dem_object.ratio)
                row[1].set_xlim(extent[:2])
                row[1].set_ylim(extent[2:])
            else:
                row[1].imshow(empty, extent=extent)

            if fs_object._metadata["bottom"]:
                row[2].imshow(dem, cmap=cmap, extent=extent)
            else:
                row[2].imshow(empty)

        elif fs_object._has_type == "__FROM_ELEVATION":

            row[1].imshow(empty, extent=extent)
            row[2].imshow(empty, extent=extent)

            discrete_dem = np.zeros(dem_object.dem.shape)
            discrete_dem.fill(np.nan)
            heights = [0] + fs_object._data
            heights.sort()

            for i in range(len(heights)):
                discrete_dem[dem_object.dem > heights[i]] = i * 10

            row[0].imshow(discrete_dem, cmap=cmap, extent=extent)

        elif fs_object._has_type == "__SIDESETS":

            row[0].imshow(empty, extent=extent)
            row[2].imshow(empty, extent=extent)

            data = fs_lib.__facesets_from_coordinates(
                {"all": fs_object._data}, dem_object.boundary
            )["all"]

            bnd = dem_object.boundary

            for i in np.unique(data):

                if i == 1 and fs_object._metadata["layers"][0] == 0:
                    continue

                mask = data == i
                row[1].scatter(bnd[:, 0][mask], bnd[:, 1][mask])

            row[1].set_aspect(dem_object.ratio)
            row[1].set_xlim(extent[:2])
            row[1].set_ylim(extent[2:])

        else:
            raise ValueError("Malformed faceset object")

    if not isinstance(fs_list, list):
        fs_list = [fs_list]

    if dem_object.boundary is None:
        raise ValueError(
            "Please generate a DEM boundary before " "running this function"
        )

    dem = deepcopy(dem_object.dem)
    dem = np.ones(dem.shape)
    dem[dem_object.dem == dem_object.no_data_value] = np.nan
    extent = dem_object.extent

    rows, cols = len(fs_list), 3

    f, axes = plt.subplots(
        rows, cols, figsize=(12, 8), sharex=True, sharey=True
    )

    top_axes_row = axes[0] if rows > 1 else axes

    top_axes_row[0].set_title("Top")
    top_axes_row[1].set_title("Sides")
    top_axes_row[2].set_title("Bottom")

    if rows > 1:
        for row in range(rows):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plot_row(fs_list[row], axes[row], extent=extent)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plot_row(fs_list[0], top_axes_row, extent=extent)

    plt.show()
