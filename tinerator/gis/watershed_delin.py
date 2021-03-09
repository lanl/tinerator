import io
from contextlib import redirect_stdout
import numpy as np
import richdem as rd
from .utils import project_vector
from .vector import Shape, ShapeType
from .raster import Raster

def watershed_delineation(
    raster: Raster,
    threshold: float,
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

    # Generate a polyline from data
    threshold_matrix = accum_matrix > threshold
    xy = np.transpose(np.where(threshold_matrix == True))
    xy[:, 0], xy[:, 1] = xy[:, 1], xy[:, 0].copy()
    xy = xy.astype(float)

    # Was threshold too high? Or method/params wrong?
    if np.size(xy) == 0:
        raise ValueError(
            "Could not generate feature. Threshold may be too high."
        )

    # Put data into Shape object
    xy = Shape(
        points=project_vector(xy, raster),
        crs=raster.crs,
        shape_type=ShapeType.POINT,
    )

    if return_matrix:
        return (xy, accum_matrix)

    return xy
