import io
from contextlib import redirect_stdout
from copy import deepcopy
import richdem as rd
import numpy as np
from .raster import Raster
from .utils import project_vector

# Should inherit from some kind of shapefile format
class Shape:
    def __init__(self, points, crs):
        self.points = points
        self.crs = crs

    def plot(self):
        pass

    def save(self):
        pass

    def reproject(self):
        pass

def watershed_delineation(
    raster: Raster,
    threshold: float,
    method: str = "D8",
    exponent:float = None,
    weights: rd.rdarray = None,
    return_matrix: bool = False,
) -> np.ndarray:
    """
    Performs watershed delination on a DEM.
    Optionally, fills DEM pits and flats.

    :param dem: richdem array
    :type dem: richdem.rdarray
    :param fill_depressions: flag to fill DEM pits / holes
    :type fill_depressions: bool
    :param fill_flats: flag to fill DEM flats
    :type fill_flats: bool
    :param method: flow direction algorithm
    :type method: string

    Returns:
    :param accum: flow accumulation matrix
    :type accum: np.ndarray
    """

    if isinstance(raster, Raster):
        elev_raster = raster.data
    else:
        raise ValueError(f"Incorrect data type for `raster`: {type(raster)}")

    f = io.StringIO()

    with redirect_stdout(f):
        accum_matrix = rd.FlowAccumulation(
            elev_raster, 
            method=method, 
            exponent=exponent, 
            weights=weights, 
            in_place=False
        )

    # Generate a polyline from data
    threshold_matrix = accum_matrix > threshold
    xy = np.transpose(np.where(threshold_matrix == True))
    xy[:, 0], xy[:, 1] = xy[:, 1], xy[:, 0].copy()
    xy = xy.astype(float)

    # 
    if np.size(xy) == 0:
        raise ValueError("Could not generate feature. Threshold may be too high.")

    xy = Shape(points = project_vector(xy, raster), crs = raster.crs)

    if return_matrix:
        return (xy, accum_matrix)

    return xy