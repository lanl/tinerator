import fiona
from distutils.version import LooseVersion
from itertools import chain
from collections import OrderedDict
import pyproj
import numpy as np
from pyproj.crs import CRS
from shapely.geometry import shape as to_shapely_shape
from shapely.geometry import mapping as shapely_mapping
from .geoutils import parse_crs
from ..visualize import plot as pl
from ..logging import log, warn, debug, error

class Geometry:
    """
    Creates a Geometry object. This object stores a collection
    of shapes in Shapely format (under ``Geometry.shapes``), along with
    a CRS and individual shape attributes.
    
    Refer to the Shapely documentation at https://shapely.readthedocs.io for 
    the various methods that Shapely objects support.
    """
    
    def __init__(self, shapes: list = None, crs: pyproj.CRS = None, properties: OrderedDict = None):
        self.crs = parse_crs(crs)
        self.shapes = shapes
        
        if properties is None:
            self.properties = OrderedDict()
        else:
            self.properties = properties
    
    def __len__(self):
        return len(self.shapes)
    
    def __str__(self):
        return f"Geometry<\"{self.geometry_type}\", shapes={len(self)}, crs=\"{self.crs.name}\">"

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
        bounds = np.array([s.bounds for s in g.shapes])

        return (
            np.min(bounds[:,0]),
            np.min(bounds[:,1]),
            np.max(bounds[:,2]),
            np.max(bounds[:,3]),
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
        return np.array([(x[0], x[1]) for x in shp.coords[:] for shp in self.shapes])
    
    def plot(
        self,
        layers: list = None,
        outfile: str = None,
        **kwargs
    ):
        """
        Plots the Shape object.
        """

        objects = [self]

        if layers is not None:
            if not isinstance(layers, list):
                layers = [layers]
            objects += layers

        pl.plot_objects(
            objects,
            outfile=outfile,
            xlabel=f"Easting ({self.units})",
            ylabel=f"Northing ({self.units})",
            **kwargs
        )

    def save(self, outfile: str, driver: str = 'ESRI Shapefile'):
        """
        Saves the Geometry object to a shapefile.
        
        Arguments
        ---------
        
        outfile (str): The path to save the Geometry object.
        driver (:obj:`str`, optional): The file format driver. May be one of:
            ``['ESRI Shapefile', 'GeoJSON']``.
        """
        attributes = [shp.properties.items() for shp in self.shapes]
        #properties = OrderedDict(chain(*attributes))
        
        gtype = self.geometry_type
        
        if driver == 'ESRI Shapefile':
            # ESRI Shapefiles do not understand these formats.
            if 'MultiLineString' in gtype:
                gtype = gtype.replace('MultiLineString', 'LineString')
            elif 'MultiPolygon' in gtype:
                gtype = gtype.replace('MultiPolygon', 'Polygon')
        
        schema = {
            'geometry': gtype,
            'properties': self.properties
        }
        
        if LooseVersion(fiona.__gdal_version__) < LooseVersion("3.0.0"):
            crs = self.crs.to_wkt(pyproj.enums.WktVersion.WKT1_GDAL)
        else:
            # GDAL 3+ can use WKT2
            crs = self.crs.to_wkt()
        
        with fiona.open(
            outfile, 'w', 
            crs_wkt=crs, 
            driver=driver, 
            schema=schema
        ) as output:
            for (i, shape) in enumerate(self.shapes):
                try:
                    props = shape.properties
                except AttributeError:
                    props = OrderedDict()

                output.write(
                    {
                        'geometry': shapely_mapping(shape),
                        'properties': props,
                        'id': str(f'{i+1}'),
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
            geom = 'GeometryCollection'
        else:
            geom = s_types[0]

        if len(self) > 1:
            if geom != 'GeometryCollection':
                geom = f'Multi{geom}'

        if self.ndim == 3:
            geom = f'3D {geom}'

        return geom
        

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
    
    with fiona.open(filename, 'r') as c:
        shapes = []
        crs = CRS.from_wkt(c.crs_wkt)
        
        for next_shape in iter(c):
            shp_shapely = to_shapely_shape(next_shape['geometry'])
            
            # TODO: properties should probably be global to the
            # Geometry class
            if 'properties' in next_shape:
                shp_shapely.properties = next_shape['properties']
            else:
                shp_shapely.properties = None
        
            shapes.append(shp_shapely)
        
        return Geometry(shapes=shapes, crs=crs, properties=c.schema['properties'])