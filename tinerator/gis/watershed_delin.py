import io
import os
import tempfile
from contextlib import redirect_stdout
import numpy as np
import richdem as rd
from scipy.spatial.distance import cdist
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge
from typing import List, Optional, Tuple
from .geoutils import project_vector
from .vector import Shape, ShapeType
from .raster import Raster
from .geometry import Geometry
from ..logging import log, warn, debug


def pysheds_watershed_delineation(
    raster: Raster,
    pour_point: List[int] = None,
    pour_point_type: str = "label",
    threshold: float = None,
    dirmap: List[int] = (64, 128, 1, 2, 4, 8, 16, 32),
    return_matrix: bool = False,
    fill_depressions: bool = False,
    resolve_flats: bool = False,
    method: str = "D8",
    recursion_limit: int = 15000,
) -> Tuple[Geometry, Optional[np.ndarray]]:
    """

    Args
    ----
        pour_point_type (:obj:`str`, optional): How to interpret parameter ``pour_point``.
            'index' : ``pour_point`` represents the column and row indices of the pour point.
            'label' : ``pour_point`` represents geographic coordinates (will be snapped to nearest cell).

        method (str): Routing algorithm to use:
            'D8' : D8 flow directions
            'dinf' : D-infinity flow directions

        recusionlimit (int): Recursion limit. May need to be raised if recursion limit is reached.
        dirmap (List[int]): List of integer values representing the following
            cardinal and intercardinal directions (in order):
                    [N, NE, E, SE, S, SW, W, NW]
    """
    from pysheds.grid import Grid
    from shapely.geometry import shape
    from .geoutils import parse_crs

    try:
        x, y = pour_point
    except (TypeError, IndexError) as e:
        raise ValueError(
            f"`pour_point` must be a tuple in the form: (x, y). "
            f"Not: {pour_point}. {e}"
        )

    method = method.lower()

    with tempfile.TemporaryDirectory() as tmp_dir:
        raster.save(os.path.join(tmp_dir, "raster.tif"))

        dem_name = "dem"
        grid = Grid.from_raster(os.path.join(tmp_dir, "raster.tif"), data_name=dem_name)
        # Grid.read_raster ????

        # Fill depressions in DEM
        if fill_depressions:
            out_dem = "flooded_dem"
            grid.fill_depressions(dem_name, out_name=out_dem)
            dem_name = out_dem

        # Resolve flats in DEM
        if resolve_flats:
            out_dem = "inflated_dem"
            grid.resolve_flats(dem_name, out_name=out_dem)
            dem_name = out_dem

        # Calculate flow direction matrix
        grid.flowdir(data=dem_name, out_name="dir", dirmap=dirmap)

        # Delineate the catchment
        grid.catchment(
            data="dir",
            x=x,
            y=y,
            dirmap=dirmap,
            out_name="catch",
            recursionlimit=recursion_limit,
            xytype=pour_point_type,
            routing=method,
        )

        grid.accumulation(data="catch", dirmap=dirmap, routing=method, out_name="acc")

        accum_matrix = np.array(grid.view("acc"))

        if threshold is None:
            threshold = int(round(np.mean(accum_matrix) + np.std(accum_matrix)))
            log(
                "Threshold was not set by user; "
                f"set automatically to: {round(threshold, 5)}. "
                "Adjust this value to adjust the river network."
            )

        # Extract river network
        branches = grid.extract_river_network(
            fdir="catch",
            acc="acc",
            threshold=int(round(threshold)),
            dirmap=dirmap,
            routing=method,
        )
        # from matplotlib import pyplot as plt
        # for branch in branches['features']:
        #    line = np.asarray(branch['geometry']['coordinates'])
        #    plt.plot(line[:, 0], line[:, 1])
        # plt.show()
        # exit()
        shapes = [shape(feature["geometry"]) for feature in branches["features"]]

        if len(shapes) == 0:
            raise ValueError("Could not generate feature. Threshold may be too high.")

        # Put data into Geometry object
        xy = Geometry(
            shapes=shapes,
            crs=parse_crs(grid.crs),
        )

        if return_matrix:
            return (xy, accum_matrix)

        return xy


def watershed_delineation(
    raster: Raster,
    threshold: float = None,
    method: str = "D8",
    exponent: float = None,
    weights: np.ndarray = None,
    return_matrix: bool = False,
) -> Tuple[Geometry, Optional[np.ndarray]]:
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
