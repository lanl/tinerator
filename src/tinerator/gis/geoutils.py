import copy
import numpy as np
import math
from scipy.spatial.distance import euclidean, cdist
from scipy import ndimage as nd
from osgeo import ogr, gdal, gdal_array
import pyproj
from pyproj import CRS
from pyproj.crs import CRSError
from ..logging import log, warn, debug, error
from ..constants import DEFAULT_PROJECTION
from typing import Any, Callable, List, Union


def parse_crs(crs: Union[str, int, dict]) -> CRS:
    """
    Returns a pyproj.CRS object from:
        * A string (Proj4 string, "epsg:xxxx", Wkt string, ...)
        * An int (EPSG code)
        * A dict with Proj4 projection

    See the Proj4 documentation for more information.

    https://pyproj4.github.io/pyproj/dev/api/crs/crs.html

    Args:
        crs (Union[str, int, dict]) : A PyProj4-supported projection.

    Returns:
        A pyproj.CRS object.

    Examples:
        >>> tin.gis.parse_crs(26915)
        >>> tin.gis.parse_crs("EPSG:26915")
        >>> tin.gis.parse_crs("+proj=geocent +datum=WGS84 +towgs84=0,0,0")
    """
    try:
        if isinstance(crs, CRS):
            return crs
        elif isinstance(crs, str):
            return CRS.from_string(crs)
        elif isinstance(crs, int):
            return CRS.from_epsg(crs)
        elif isinstance(crs, dict):
            return CRS.from_dict(crs)
        elif isinstance(crs, pyproj.Proj):
            return crs.crs
        else:
            return CRS.from_user_input(crs)
    except CRSError:
        warn(f'Could not parse CRS "{crs}". Defaulting to "{DEFAULT_PROJECTION}"')
        return CRS.from_string(DEFAULT_PROJECTION)


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
    ind = nd.distance_transform_edt(
        invalid, return_distances=False, return_indices=True
    )
    data = data[tuple(ind)]

    # --- END INTERPOLATING DEM DATA ---- #

    n_nodes = nodes.shape[0]
    z_array = np.zeros((n_nodes,), dtype=float)

    indices = unproject_vector(nodes, dem)

    x_idx = indices[:, 0]
    y_idx = indices[:, 1]

    try:
        for i in range(n_nodes):
            z_array[i] = data[y_idx[i]][x_idx[i]]
    except IndexError as e:
        raise IndexError(
            "Mesh nodes are out of bounds of raster. Try reprojecting. {e}"
        )

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

    map_x = lambda x: (cellSize + 2.0 * float(x) - 2.0 * xllCorner) / (2.0 * cellSize)
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
        yllCorner + (float(0.0 - y + float(nRows)) * cellSize) - (cellSize / 2.0)
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
    xy = np.transpose(np.where(threshold_matrix is True))
    xy[:, 0], xy[:, 1] = xy[:, 1], xy[:, 0].copy()

    return xy


def order_points(points: np.ndarray, opt: str = "polar", clockwise: bool = True):
    """
    Given a 2D array of points, this function reorders points clockwise.
    Available methods are: 'angle', to sort by angle, 'polar', to sort by
    polar coordinates, and 'nearest_neighbor', to sort by nearest neighbor.

    # Arguments
    points (np.ndarray): Array of unsorted points
    opt (str): Sorting method
    clockwise (bool): order points clockwise or counterclockwise

    # Returns
    Sorted points
    """

    origin = np.mean(points, axis=0)
    refvec = [0, 1]

    def clockwise_angle_and_distance(point):
        """
        Returns angle and length from origin.
        Used as a sorting function to order points by angle.

        Author credit to MSeifert.
        """

        vector = [point[0] - origin[0], point[1] - origin[1]]
        lenvector = math.hypot(vector[0], vector[1])

        if lenvector == 0:
            return -math.pi, 0.0

        normalized = [vector[0] / lenvector, vector[1] / lenvector]
        dotprod = normalized[0] * refvec[0] + normalized[1] * refvec[1]
        diffprod = refvec[1] * normalized[0] - refvec[0] * normalized[1]

        angle = math.atan2(diffprod, dotprod)

        if angle < 0:
            return 2 * math.pi + angle, lenvector

        return angle, lenvector

    def polar_sort(point):
        return math.atan2(point[1] - origin[1], point[0] - origin[0])

    def nearest_neighbor_sort(xy: np.ndarray):
        dist_matrix = cdist(xy, xy, "euclidean")
        nil_value = np.max(dist_matrix) + 1000
        mapper = np.empty((np.shape(xy)[0],), dtype=int)

        count = 0
        indx = 0
        while count < np.shape(mapper)[0]:
            dist_matrix[indx, :] = nil_value
            indx = np.argmin(dist_matrix[:, indx])
            mapper[count] = indx
            count += 1

        return xy[mapper]

    if opt.lower() == "polar":
        pts = np.array(sorted(points, key=clockwise_angle_and_distance))
    elif opt.lower() == "angle":
        pts = np.array(sorted(points, key=polar_sort))
    elif opt.lower() == "nearest_neighbor":
        pts = nearest_neighbor_sort(points)
    else:
        raise ValueError("Unknown sorting method")

    if not clockwise:
        pts = pts[::-1]

    return pts
