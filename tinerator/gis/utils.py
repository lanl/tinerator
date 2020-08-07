import copy
import numpy as np
from scipy import ndimage as nd
from osgeo import ogr, gdal, gdal_array


def map_elevation(dem, nodes: np.ndarray) -> np.ndarray:
    """
    Maps elevation from a DEM raster to mesh nodes.
    """

    array = dem.masked_data()

    # --- BEGIN INTERPOLATING DEM DATA ---- #
    # This is done to keep the z_value indexing from landing on
    # NaNs.
    data = copy.copy(array)
    invalid = np.isnan(data)
    ind = nd.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    data = data[tuple(ind)]

    # --- END INTERPOLATING DEM DATA ---- #

    n_nodes = nodes.shape[0]
    z_array = np.zeros((n_nodes,), dtype=float)

    indices = unproject_vector(nodes, dem)

    x_idx = indices[:, 0]
    y_idx = indices[:, 1]

    for i in range(n_nodes):
        z_array[i] = data[y_idx[i]][x_idx[i]]

    return z_array


def unproject_vector(vector: np.ndarray, raster) -> np.ndarray:
    """
    Converts a vector of (x,y) point in a particular raster's CRS back into
    [row, col] indices relative to that raster.
    """

    # TODO: verify that `vector == unproject_vector(project_vector(vector))`

    nNodes = vector.shape[0]
    xllCorner = raster.xll_corner
    yllCorner = raster.yll_corner
    cellSize = raster.cell_size
    nRows = raster.nrows

    map_x = lambda x: (cellSize + 2.0 * float(x) - 2.0 * xllCorner) / (
        2.0 * cellSize
    )
    map_y = lambda y: ((yllCorner - y) / cellSize + nRows + 1.0 / 2.0)

    x_arr = np.reshape(list(map(map_x, vector[:, 0])), (nNodes, 1))
    y_arr = np.reshape(list(map(map_y, vector[:, 1])), (nNodes, 1))

    return np.hstack((np.round(x_arr), np.round(y_arr))).astype(int) - 1


def project_vector(vector: np.ndarray, raster) -> np.ndarray:
    """
    Because every raster has a CRS projection, associated indices
    in that raster can be projected into that coordinate space.

    For example, imagine a DEM. The pixel at index [0,0] corresponds to
    (xll_corner, yll_corner).
    """

    # TODO: something is (slightly) wrong with this calculation

    nNodes = vector.shape[0]
    xllCorner = raster.xll_corner
    yllCorner = raster.yll_corner
    cellSize = raster.cell_size
    nRows = raster.nrows

    map_x = lambda x: (xllCorner + (float(x) * cellSize) - (cellSize / 2.0))
    map_y = lambda y: (
        yllCorner
        + (float(0.0 - y + float(nRows)) * cellSize)
        - (cellSize / 2.0)
    )

    x_arr = np.reshape(list(map(map_x, vector[:, 0])), (nNodes, 1))
    y_arr = np.reshape(list(map(map_y, vector[:, 1])), (nNodes, 1))

    if vector.shape[1] > 2:
        return np.hstack((x_arr, y_arr, np.reshape(vector[:, 2], (nNodes, 1))))

    return np.hstack((x_arr, y_arr))


def rasterize_shapefile_like(
    shpfile: str, model_raster_fname: str, nodata_val: float = 0
):
    """
    Given a shapefile, rasterizes it so it has
    the exact same extent as the given model_raster
    Taken from [0].
    [0]: https://github.com/terrai/rastercube/blob/master/rastercube/datasources/shputils.py
    """

    dtype = gdal.GDT_Float64

    model_dataset = gdal.Open(model_raster_fname)
    shape_dataset = ogr.Open(shpfile)
    shape_layer = shape_dataset.GetLayer()
    mem_drv = gdal.GetDriverByName("MEM")
    mem_raster = mem_drv.Create(
        "", model_dataset.RasterXSize, model_dataset.RasterYSize, 1, dtype
    )
    mem_raster.SetProjection(model_dataset.GetProjection())
    mem_raster.SetGeoTransform(model_dataset.GetGeoTransform())
    mem_band = mem_raster.GetRasterBand(1)
    mem_band.Fill(nodata_val)
    mem_band.SetNoDataValue(nodata_val)

    err = gdal.RasterizeLayer(mem_raster, [1], shape_layer, None, None, [1])

    assert err == gdal.CE_None, "Could not rasterize layer"
    return mem_raster.ReadAsArray()


def get_feature_trace(
    feature: np.ndarray, feature_threshold: float = 750.0
) -> np.ndarray:
    """
    Returns an array of (x,y) pairs corresponding to values over a given
    threshold in a feature array.
    :param feature: 
    :type feature:
    :param distance:
    :type distance:
    :param feature_threshold:
    :type feature_threshold:
    :returns: 
    """

    threshold_matrix = feature > feature_threshold
    xy = np.transpose(np.where(threshold_matrix == True))
    xy[:, 0], xy[:, 1] = xy[:, 1], xy[:, 0].copy()

    return xy
