import io
from contextlib import redirect_stdout
from copy import deepcopy
import richdem as rd
import numpy as np
import shapefile
import glob
import os
from enum import Enum, auto
import pyproj
from pyproj import CRS
from pyproj.crs import CRSError
from .utils import project_vector, parse_crs
from ..visualize import plot as pl
from ..logging import log, warn, debug, error


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

        if self.connectivity is not None and self.shape_type == ShapeType.POINT:
            error("Shape of type `point` has meaningless connectivity.")

        self.crs = parse_crs(crs)

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
            title = (
                f'Shape: "{os.path.basename(self.filename)}" | CRS: "{self.crs.name}"'
            )

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
        crs_outfile = os.path.splitext(filename)[0] + ".prj"

        with shapefile.Writer(filename) as w:

            debug(f"Attempting to save vector object to {filename}")

            # MULTIPOINT
            if self.shape_type == ShapeType.POINT:
                w.field("name", "C")
                w.multipoint(points)
                w.record("multipoint1")

            # LINESTRING
            elif self.shape_type == ShapeType.POLYLINE:
                warn(
                    "Saving shapefile as polyline: does not take connectivity vector into account"
                )
                w.field("name", "C")
                w.line([points])
                w.record("linestring1")

            # POLYGON
            elif self.shape_type == ShapeType.POLYGON:
                warn(
                    "Saving shapefile as polygon: does not take connectivity vector into account"
                )
                w.field("name", "C")
                w.poly([points])
                w.record("polygon1")

            else:
                raise ValueError(f"Unknown shape_type: {self.shape_type}")

        log(f"Shapefile data written to {filename}")

        # CRS info must be written out manually. See the reader.
        with open(crs_outfile, "w") as f:
            f.write(self.crs.to_wkt(version=pyproj.enums.WktVersion.WKT1_ESRI))

        log(f"CRS information written to {crs_outfile}")

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
