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
from .utils import project_vector
from ..visualize import plot as pl


class ShapeType(Enum):
    POINT = auto()
    POLYLINE = auto()
    POLYGON = auto()


class Shape:
    def __init__(
        self,
        points: np.ndarray,
        crs: str,
        shape_type: ShapeType,
        filename: str = "",
        connectivity: np.ndarray = None,
    ):
        self.filename = filename
        self.points = points
        self.connectivity = connectivity
        self.shape_type = shape_type

        if (
            self.connectivity is not None
            and self.shape_type == ShapeType.POINT
        ):
            error("Shape of type `point` has meaningless connectivity.")

        try:
            self.crs = CRS.from_wkt(crs)
        except (CRSError, TypeError):
            print("Could not parse CRS. Defaulting to EPSG: 32601.")
            self.crs = CRS.from_epsg(32601)

    def __repr__(self):
        display = "\ntinerator.gis.Shape object\n"
        display += "=" * 30 + "\n"
        display += "Source\t\t: file (%s)\n" % self.filename
        display += "Shape type\t: %s\n" % repr(self.shape_type)
        display += "Points\t\t: %s\n" % repr(len(self.points))
        display += "CRS\t\t: %s\n" % self.crs.name
        display += "Units\t\t: %s\n" % self.units
        display += "Extent\t\t: %s\n" % repr(self.extent)
        display += "\n%s\n" % repr(self.points)

        return display

    @property
    def units(self):
        return self.crs.axis_info[0].unit_name

    @property
    def centroid(self):
        """
        Returns the shape centroid.
        """
        xmin, ymin, xmax, ymax = self.extent
        return (xmin + (xmax - xmin) / 2.0, ymin + (ymax - ymin) / 2.0)

    @property
    def extent(self):
        """
        Returns the spatial extent of the shape `(xmin, ymin, xmax, ymax)`.
        """

        return (
            np.nanmin(self.points[:, 0]),
            np.nanmin(self.points[:, 1]),
            np.nanmax(self.points[:, 0]),
            np.nanmax(self.points[:, 1]),
        )

    def plot(
        self,
        layers: list = None,
        outfile: str = None,
        title: str = None,
        raster_hillshade=False,
    ):
        """
        Plots the Shape object.
        """

        if title is None:
            title = f'Shape: "{os.path.basename(self.filename)}" | CRS: "{self.crs.name}"'

        objects = [self]

        if layers is not None:
            if not isinstance(layers, list):
                layers = [layers]
            objects += layers

        pl.plot_objects(
            objects,
            outfile=outfile,
            title=title,
            xlabel=f"Easting ({self.units})",
            ylabel=f"Northing ({self.units})",
            raster_hillshade=raster_hillshade,
        )

    def save(self, filename: str):
        """
        Saves shape object as an ESRI Shapefile.
        """

        # Convert points to list
        points = self.points.tolist()

        # Get CRS in WKT string format
        crs = self.crs.to_wkt()  # pretty=True)
        crs_outfile = os.path.splitext(filename)[0] + ".prj"

        with shapefile.Writer(filename) as w:

            print(self.shape_type)
            print(self.points)

            """MULTIPOINT"""
            if self.shape_type == ShapeType.POINT:
                w.field("name", "C")
                w.multipoint(points)
                w.record("multipoint1")
                print("Saved")

            """LINESTRING"""
            if self.shape_type == ShapeType.POLYLINE:
                raise NotImplementedError("sorry!")
                w.field("name", "C")
                w.line(
                    [
                        [[1, 5], [5, 5], [5, 1], [3, 3], [1, 1]],  # line 1
                        [[3, 2], [2, 6]],  # line 2
                    ]
                )
                w.record("linestring1")

            """POLYGON"""
            if self.shape_type == ShapeType.POLYGON:
                raise NotImplementedError("sorry!")
                w.field("name", "C")
                # Polygon points must be ordered clockwise
                w.poly(
                    [
                        [
                            [113, 24],
                            [112, 32],
                            [117, 36],
                            [122, 37],
                            [118, 20],
                        ],  # poly 1
                        [[116, 29], [116, 26], [119, 29], [119, 32]],  # hole 1
                        [[15, 2], [17, 6], [22, 7]],  # poly 2
                    ]
                )
                w.record("polygon1")

        # CRS info must be written out manually. See the reader.
        with open(crs_outfile, "w") as f:
            f.write(crs)

    def reproject(self, to_crs: str):
        # See tin.gis.reproject_shapefile
        # Change self.crs
        # https://pyproj4.github.io/pyproj/dev/api/crs/crs.html
        print(f"Projecting to {to_crs}...jk, this is not implemented yet.")


def load_shapefile(filename: str, to_crs: str = None) -> list:
    """
    Given a path to a shapefile, reads and returns each object
    in the shapefile as a tin.gis.Shape object.
    """
    shapes = []

    type_dict = {
        shapefile.POLYGON: ShapeType.POLYGON,
        shapefile.POINT: ShapeType.POINT,
        shapefile.POLYLINE: ShapeType.POLYLINE,
    }

    if to_crs is not None:
        raise ValueError("Not supported.")

    # The projection is stored in the *.prj file as WKT format.
    # Attempt to read it.
    crs = ""
    for file in glob.glob(os.path.splitext(filename)[0] + "*"):
        if "prj" in file.lower():
            with open(file, "r") as f:
                crs = f.read()
            break

    # Read each shape from the shapefile and store in a Shape object
    with shapefile.Reader(filename) as shp:
        for shape_record in shp:
            shape = shape_record.shape

            try:
                shape_type = type_dict[shape.shapeType]
            except KeyError:
                print("WARNING: couldn't parse shape type. Skipping shape.")
                continue

            shapes.append(
                Shape(
                    points=np.array(shape.points),
                    crs=crs,
                    shape_type=shape_type,
                    filename=filename,
                )
            )

    if to_crs is not None:
        for shape in shapes:
            shape.reproject(crs)

    if len(shapes) == 1:
        return shapes[0]

    return shapes


def watershed_delineation(
    raster,  #: Raster,
    threshold: float,
    method: str = "D8",
    exponent: float = None,
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
