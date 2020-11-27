import io
from contextlib import redirect_stdout
from copy import deepcopy
import richdem as rd
import numpy as np
import shapefile
import glob
import os
from enum import Enum, auto
from pyproj import CRS
from pyproj.crs import CRSError
from .raster import Raster
from .utils import project_vector
from ..visualize import plot as pl

class ShapeType(Enum):
    POINT = auto()
    POLYLINE = auto()
    POLYGON = auto()

# Should inherit from some kind of shapefile format
class Shape:
    def __init__(self, points: np.ndarray, crs: str, shape_type: ShapeType, filename: str = None):
        self.filename = filename
        self.points = points
        self.shape_type = shape_type

        try:
            self.crs = CRS.from_wkt(crs)
        except (CRSError, TypeError):
            print("Could not parse CRS. Defaulting to EPSG: 32601.")
            self.crs = CRS.from_epsg(32601)

    def __repr__(self):
        # DEM information
        display  = "\ntinerator.gis.Shape object\n"
        display += "="*30 + "\n"
        display += "Source\t\t: file (%s)\n" % self.filename
        display += "Shape type\t: %s\n" % repr(self.shape_type)
        display += "Points\t\t: %s\n" % repr(len(self.points))
        display += "CRS\t\t: %s\n" % self.crs.name
        display += "Units\t\t: %s\n" % self.units
        display += "\n%s\n" % repr(self.points)

        return display

    @property
    def units(self):
        return self.crs.axis_info[0].unit_name

    @property
    def centroid(self):
        return (None, None)

    @property
    def bbox(self):
        return (None, None, None, None)

    def plot(self):
        '''
        Plots the object.
        '''
        pl.plot_objects([self])

    def save(self, filename: str):
        '''
        Saves shape object as an ESRI Shapefile.
        '''
        pass

    def reproject(self, to_crs: str):
        print(f'Projecting to {to_crs}...jk, this is not implemented yet.')

def load_shapefile(filename: str, to_crs: str = None) -> list:
    '''
    Given a path to a shapefile, reads and returns each object
    in the shapefile as a tin.gis.Shape object.
    '''
    shapes = []

    type_dict = {
        shapefile.POLYGON: ShapeType.POLYGON,
        shapefile.POINT: ShapeType.POINT,
        shapefile.POLYLINE: ShapeType.POLYLINE
    }

    if to_crs is not None:
        raise ValueError("Not supported.")

    # The projection is stored in the *.prj file as WKT format.
    # Attempt to read it.
    crs = ''
    for file in glob.glob(os.path.splitext(filename)[0] + '*'):
        if 'prj' in file.lower():
            with open(file, 'r') as f:
                crs = f.read()
            break

    # Read each shape from the shapefile and store in a Shape object
    with shapefile.Reader(filename) as shp:
        for shape_record in shp:
            shape = shape_record.shape

            try:
                shape_type = type_dict[shape.shapeType]
            except KeyError:
                print('WARNING: couldn\'t parse shape type. Skipping shape.')
                continue

            shapes.append(
                Shape(
                    points=np.array(shape.points),
                    crs=crs,
                    shape_type=shape_type,
                    filename=filename
                )
            )

    if to_crs is not None:
        for shape in shapes:
            shape.reproject(crs)

    if len(shapes) == 1:
        return shapes[0]

    return shapes


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