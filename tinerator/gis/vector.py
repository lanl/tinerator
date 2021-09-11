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
from .geoutils import project_vector, parse_crs
from ..visualize import plot2d, MapboxStyles
from ..logging import log, warn, debug, error


class Geometry:
    def __init__(self):
        self.shapes = []
        self.crs = None


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

    def __str__(self):
        return f'Shape<shape_type={self.shape_type}, points={len(self.points)}, CRS="{self.crs.name}">'

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
        mapbox_style: str = MapboxStyles.OPEN_STREET_MAP,
        show_legend: bool = False,
        raster_cmap: list = None,
        **kwargs,
    ):
        """
        Plots the vector object.
        """
        objects = [self]

        if layers is not None:
            if not isinstance(layers, list):
                layers = [layers]
            objects += layers

        plot2d(
            objects,
            mapbox_style=mapbox_style,
            show_legend=show_legend,
            raster_cmap=raster_cmap,
            **kwargs,
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
    in the shapefile as a :obj:`tinerator.gis.Shape` object.

    Args:
        filename (str): The input filename.
        to_crs (:obj:`str`, optional): If provided, will reproject
            the shapefile object to the given CRS (can be an EPSG code,
            WKT string, or :obj:`pyproj.CRS` object).

    Returns:
        A :obj:`tinerator.gis.Shape` object.

    Examples:
        >>> boundary = tin.gis.load_shapefile("my_shapefile.shp", crs="EPSG:3114")
    """
    shapes = []

    type_dict = {
        shapefile.POLYGON: ShapeType.POLYGON,
        shapefile.POINT: ShapeType.POINT,
        shapefile.POLYLINE: ShapeType.POLYLINE,
        11: ShapeType.POINT,  # TODO: wrong!
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
                import ipdb

                # ipdb.set_trace()
                warn(
                    f"WARNING: couldn't parse shape type {shape.shapeType}. Skipping shape."
                )
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
