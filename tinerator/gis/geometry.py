import fiona
from distutils.version import LooseVersion
from typing import Any, Callable, List, Union, NoReturn
from collections import OrderedDict
import pyproj
import numpy as np
from pyproj.crs import CRS
from shapely.ops import transform
from shapely.geometry import LineString
from shapely.geometry import shape as to_shapely_shape
from shapely.geometry import mapping as shapely_mapping
from .geoutils import parse_crs
from ..visualize import plot2d, MapboxStyles
from ..logging import log, warn, debug, error

try:
    from osgeo import gdal
except ImportError:
    import gdal

try:
    from osgeo import osr
except ImportError:
    import osr

try:
    from osgeo import ogr
except ImportError:
    import ogr


class Geometry:
    """
    Creates a Geometry object. This object stores a collection
    of shapes in Shapely format (under ``Geometry.shapes``), along with
    a CRS and individual shape attributes.

    Refer to the Shapely documentation at https://shapely.readthedocs.io for
    the various methods that Shapely objects support.
    """

    def __init__(
        self,
        shapes: list = None,
        crs: pyproj.CRS = None,
        properties: OrderedDict = None,
    ):
        self.crs = parse_crs(crs)
        self.shapes = shapes

        if properties is None:
            self.properties = {"properties": {}, "metadata": {}}
            self.properties["metadata"]["schema"] = OrderedDict({})
        else:
            self.properties = properties

    def __len__(self):
        return len(self.shapes)

    def __str__(self):
        n_props = len(self.properties["metadata"]["schema"])
        return f'Geometry<"{self.geometry_type}", shapes={len(self)}, crs="{self.crs.name}", properties={n_props}>'

    def __repr__(self):
        return str(self)

    @property
    def ndim(self):
        """
        The number of dimensions of the Geometry object. May be 2 or 3.
        """
        return np.max([s._ndim for s in self.shapes])

    @property
    def extent(self):
        """
        Returns the spatial extent of the Geometry object
        in the form ``(xmin, ymin, xmax, ymax)``.
        """
        bounds = np.array([s.bounds for s in self.shapes])

        return (
            np.min(bounds[:, 0]),
            np.min(bounds[:, 1]),
            np.max(bounds[:, 2]),
            np.max(bounds[:, 3]),
        )

    @property
    def centroid(self):
        """
        Returns the centroid of the Geometry object.
        """
        centroid_pts = [(s.centroid.x, s.centroid.y) for s in self.shapes]
        return np.mean(centroid_pts, axis=0)

    @property
    def units(self):
        """
        Returns the type of unit the CRS is in.
        """
        return self.crs.axis_info[0].unit_name

    @property
    def coordinates(self):
        """Returns the (x, y) array of all shapes in the Geometry object."""
        coords = []
        for shp in self.shapes:
            for x in shp.coords[:]:
                coords.append((x[0], x[1]))

        return np.array(coords)

    def add_property(self, name: str, data: list, type: str = None):
        """
        Adds a property to the Geometry object.

        Args:
            name (str): The name of the property.
            data (:obj:`list`): A list containing the property data. Must be of length ``len(self.shapes)``.
            type (:obj:`str`, optional): The type of the data. Must be `str`, `int`, `float`, or `date`.
        """
        assert len(data) == len(
            self.shapes
        ), "`data` must be a list equal in length to self.shapes"
        if type is None:
            mapper = {int: "int", float: "float", str: "str"}

            try:
                type = mapper[type(data[0])]
            except KeyError:
                raise AttributeError(
                    f"Could not automatically parse type {type(data[0])}. Pass `type = ...` and try again."
                )

        self.properties["metadata"]["schema"][name] = type
        self.properties["properties"][name] = data

    def plot(
        self,
        layers: list = None,
        mapbox_style: str = MapboxStyles.OPEN_STREET_MAP,
        show_legend: bool = False,
        raster_cmap: list = None,
        **kwargs,
    ):
        """
        Plots the Geometry object.
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

    def save(self, outfile: str, driver: str = "ESRI Shapefile"):
        """
        Saves the Geometry object to a shapefile.

        Args:
            outfile (str): The path to save the Geometry object.
            driver (:obj:`str`, optional): The file format driver. May be one of: ``['ESRI Shapefile', 'GeoJSON']``.
        """
        gtype = self.geometry_type

        if driver == "ESRI Shapefile":
            # ESRI Shapefiles do not understand these formats.
            if "MultiLineString" in gtype:
                gtype = gtype.replace("MultiLineString", "LineString")
            elif "MultiPolygon" in gtype:
                gtype = gtype.replace("MultiPolygon", "Polygon")
            elif "MultiPoint" in gtype:
                gtype = gtype.replace("MultiPoint", "Point")

        properties = self.properties["properties"]
        property_schema = self.properties["metadata"]["schema"]
        schema = {"geometry": gtype, "properties": property_schema}

        if LooseVersion(fiona.__gdal_version__) < LooseVersion("3.0.0"):
            crs = self.crs.to_wkt(pyproj.enums.WktVersion.WKT1_GDAL)
        else:
            # GDAL 3+ can use WKT2
            crs = self.crs.to_wkt()

        with fiona.open(
            outfile, "w", crs_wkt=crs, driver=driver, schema=schema
        ) as output:
            for (i, shape) in enumerate(self.shapes):

                props = OrderedDict({})
                for key in property_schema:
                    value = properties[key][i]

                    if value is None:
                        props[key] = None
                        continue

                    prop_type = property_schema[key]
                    if "int" in prop_type:
                        value = int(value)
                    elif "float" in prop_type:
                        value = float(value)
                    elif "date" in prop_type:
                        value = str(value)
                    elif "str" in prop_type:
                        value = str(value)
                    else:
                        raise ValueError(f"Could not parse type of: {key}:{prop_type}")

                    props[key] = value

                output.write(
                    {
                        "geometry": shapely_mapping(shape),
                        "properties": props,
                        "id": str(f"{i+1}"),
                    }
                )

    @property
    def geometry_type(self):
        """
        Returns the geometry type of this object in GeoJSON
        format.
        """
        s_types = np.unique([shp.type for shp in self.shapes])

        if len(s_types) > 1:
            geom = "GeometryCollection"
        else:
            geom = s_types[0]

        if len(self) > 1:
            if geom != "GeometryCollection":
                geom = f"Multi{geom}"

        if self.ndim == 3:
            geom = f"3D {geom}"

        return geom

    def polygon_exterior(self, shape_index: int = 0, spacing: float = None):
        """
        Returns the exterior nodes and connectivity
        arrays for a polygon geometry.

        Args
        ----
            shape_index (:obj:`int`, optional): Computes the exterior of the
                ``self.shapes[shape_index]`` shape.

        Returns
        -------
            Coordinates and connectivity for exterior linear ring
        """

        try:
            exterior = self.shapes[shape_index].exterior
        except AttributeError as e:
            raise AttributeError(f"Incorrect shape type; Polygon required. {e}")

        if spacing is None:
            exterior_interp = LineString(exterior)
        else:
            distances = np.arange(0, exterior.length, spacing)
            exterior_interp = LineString(
                [exterior.interpolate(distance) for distance in distances]
            )

        return Geometry(shapes=[exterior_interp], crs=self.crs, properties=None)

    def reproject(
        self, crs: Union[pyproj.CRS, str, int, dict], in_place: bool = False
    ) -> Union["Geometry", NoReturn]:
        """
        Reprojects a Geometry object into the provided CRS and returns
        the new object.

        Args:
            shape (Geometry): The Geometry object to reproject.
            crs (Union[pyproj.CRS, str, int, dict]): The CRS to reproject into.

        Returns:
            If ``in_place == False``, a TINerator Geometry object in the new coordinate reference space.
        """

        crs = parse_crs(crs)
        project = pyproj.Transformer.from_crs(self.crs, crs)
        new_shapes = [transform(project.transform, shp) for shp in self.shapes]

        return Geometry(shapes=new_shapes, crs=crs, properties=self.properties)


def load_shapefile(filename: str) -> Geometry:
    """
    Given a path to a shapefile, reads and returns each object
    in the shapefile as a :obj:`tinerator.gis.Shape` object.

    Args:
        filename (str): The input filename.

    Returns:
        A :obj:`tinerator.gis.Geometry` object.

    Examples:
        >>> boundary = tin.gis.load_shapefile("my_shapefile.shp", crs="EPSG:3114")
    """

    with fiona.open(filename, "r") as c:
        shapes = []

        properties = {"properties": [], "metadata": {}}
        properties["metadata"]["schema"] = c.schema["properties"]

        properties["properties"] = {key: [] for key in c.schema["properties"]}

        crs = CRS.from_wkt(c.crs_wkt)

        for next_shape in iter(c):
            shp_shapely = to_shapely_shape(next_shape["geometry"])

            shp_props = next_shape["properties"]

            for key in shp_props:
                properties["properties"][key].append(shp_props[key])

            shapes.append(shp_shapely)

        return Geometry(shapes=shapes, crs=crs, properties=properties)
