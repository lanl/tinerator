import io
from contextlib import redirect_stdout
import numpy as np
import richdem as rd
from scipy.spatial.distance import cdist
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge
from .geoutils import project_vector
from .vector import Shape, ShapeType
from .raster import Raster
from .geometry import Geometry
from ..logging import log, warn, debug


def watershed_delineation(
    raster: Raster,
    threshold: float = None,
    method: str = "D8",
    exponent: float = None,
    weights: np.ndarray = None,
    return_matrix: bool = False,
) -> np.ndarray:
    """
    Performs watershed delination on a DEM.
    Optionally, fills DEM pits and flats.

    Parameters
    ----------

    raster : tin.gis.Raster object
        A DEM as a TINerator Raster object.

    threshold : float, optional
        The numerical threshold where every cell with a value
        above this number is considered a flow path, and every
        cell under this number is not.

    method : str, optional
       The watershed delineation algorithm, one of:
       * D8,

    Returns
    -------
    river_network : tin.gis.Shape
       The extracted river network.

    Examples
    --------
    >>> d = tin.gis.load_raster("dem.tif")
    >>> rn = tin.gis.watershed_delineation(d, 500.)
    >>> d.plot(layers=[rn])

    """

    # if isinstance(raster, Raster):
    #    elev_raster = raster.data
    # else:
    #    raise ValueError(f"Incorrect data type for `raster`: {type(raster)}")
    elev_raster = raster.data

    f = io.StringIO()
    with redirect_stdout(f):
        accum_matrix = rd.FlowAccumulation(
            elev_raster,
            method=method,
            exponent=exponent,
            weights=weights,
            in_place=False,
        )

    if threshold is None:
        M = np.asarray(accum_matrix)
        threshold = np.mean(M) + np.std(M)
        log(
            f"Threshold was not set by user; set automatically to: {round(threshold, 5)}. Adjust this value to adjust the river network."
        )

    # Generate a polyline from data
    threshold_matrix = accum_matrix > threshold
    xy = np.transpose(np.where(threshold_matrix == True))
    xy[:, 0], xy[:, 1] = xy[:, 1], xy[:, 0].copy()
    xy = xy.astype(float)

    # Was threshold too high? Or method/params wrong?
    if np.size(xy) == 0:
        raise ValueError("Could not generate feature. Threshold may be too high.")

    # At this point, we just have a collection of points
    # that compose the delineation.
    # Below, we turn it into a MultiLineString object.
    dist_matrix = cdist(xy, xy)
    conn = np.argwhere((dist_matrix < 1.5) & (dist_matrix > 0))
    conn = np.unique(np.sort(conn), axis=0)
    multiline = linemerge(MultiLineString(xy[conn].tolist()))

    lines = []
    for line in multiline:
        coords = np.array(line.coords[:])
        coords = project_vector(coords, raster)
        lines.append(LineString(coords))

    # Put data into Geometry object
    xy = Geometry(
        shapes=lines,
        crs=raster.crs,
    )

    if return_matrix:
        return (xy, accum_matrix)

    return xy