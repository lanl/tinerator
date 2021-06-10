import richdem as rd
import numpy as np
import io
import os
import pyproj
import tempfile
from contextlib import redirect_stdout
from copy import deepcopy
from pyproj import CRS
from pyproj.crs import CRSError
from shapely.ops import polygonize, linemerge
from ..logging import log, warn, debug, error
from .geometry import Geometry
from ..visualize import plot as pl
from .geoutils import project_vector, unproject_vector, parse_crs
from .raster_boundary import square_trace_boundary as st_boundary
from ..constants import DEFAULT_NO_DATA_VALUE, DEFAULT_PROJECTION

try:
    from osgeo import gdal
except ImportError:
    import gdal

# Rendering a DEM in 3D:
# https://pvgeo.org/examples/grids/read-esri.html#sphx-glr-examples-grids-read-esri-py

extension = lambda x: os.path.splitext(x)[-1].replace(".", "").lower().strip()


def load_raster(filename: str, no_data: float = None):
    """
    Loads a raster from a given filename.
    Supports all raster datatypes that GDAL supports, including
    GeoTIFF, IMG, and ASC.

    Args:
        filename (str): The input filename.
        no_data (:obj:`float`, optional): If NoData value is not encoded in the raster,
            it can be manually set here.
        to_crs (:obj:`str`, optional): If provided, will reproject raster to the given
            CRS (can be an EPSG code, WKT string, or `pyproj.CRS` object.)

    Returns:
        A `tinerator.Raster` object.

    Examples:
        >>> dem = tin.gis.load_raster("my_dem.tif", no_data=-9999., crs="EPSG:3114")
    """

    return Raster(filename, no_data=no_data)


def new_raster(
    data: np.ndarray,
    geotransform: tuple = None,
    crs: CRS = DEFAULT_PROJECTION,
    no_data: float = DEFAULT_NO_DATA_VALUE,
):
    """
    Creates a new Raster object from an NxM Numpy array.

    https://gdal.org/tutorials/geotransforms_tut.html

    Args:
        data (np.ndarray): An NxM matrix to create the raster from.
        geotransform (:obj:`tuple`, optional): The geotransform of the raster. See the GDAL
            documentation for more information.
        crs (:obj:`pyproj.CRS`, optional): The CRS to create the raster into.
        no_data (:obj:`float`, optional): The No Data value to use for the raster.

    Returns:
        A TINerator raster object.

    Examples:
        >>> A = np.array([[1,2,3],[4,5,6],[7,8,9]], dtype=float)
        >>> raster = tin.gis.new_raster(A)
    """

    data = np.array(data, dtype=np.float64)
    nrows, ncols = data.shape
    dtype = gdal.GDT_Float64

    if geotransform is None:
        xmin = 0.0
        cell_size = 1.0

        geotransform = (
            xmin,
            cell_size,
            0.0,
            float(nrows),
            0.0,
            -cell_size,
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        outfile = os.path.join(tmp_dir, "tmp_raster.tif")

        driver = gdal.GetDriverByName("GTiff")
        outdata = driver.Create(outfile, ncols, nrows, 1, dtype)
        outdata.SetGeoTransform(geotransform)
        outdata.SetProjection(crs.to_wkt(version=pyproj.enums.WktVersion.WKT1_GDAL))
        outdata.GetRasterBand(1).WriteArray(data)
        outdata.GetRasterBand(1).SetNoDataValue(no_data)
        outdata.FlushCache()
        outdata = None

        return load_raster(outfile)


class Raster:
    """
    The main Raster class object in TINerator.
    This stores the data and metadata for a loaded or created
    Raster object (like from reading in a GeoTIFF), and has
    many helper functions for reprojection, resampling,
    depression filling, and more.
    """

    def __init__(self, raster_path: str, no_data: float = None):
        self.data = rd.LoadGDAL(raster_path, no_data=no_data)
        self.no_data_value = self.data.no_data
        self.cell_size = self.data.geotransform[1]
        self.xll_corner = self.data.geotransform[0]
        self.yll_corner = self.data.geotransform[3] - self.nrows * self.cell_size
        self.filename = raster_path
        self.crs = parse_crs(self.data.projection)

    def __lt__(self, other):
        return self.data < other

    def __gt__(self, other):
        return self.data > other

    def __getitem__(self, idx):
        return self.data[idx]

    def __setitem__(self, idx, value):
        self.data[idx] = value

    def __repr__(self):
        return str(self)

    def __str__(self):
        extent = [round(x, 5) for x in self.extent]
        return f'Raster<data.shape={self.data.shape}, extent={extent}, CRS="{self.crs.name}">'

    @property
    def geotransform(self):
        """
        Raster geotransform.
        In the form of:
        (x_min, pixel_width, 0, y_min, 0, pixel_width)
        """
        return self.data.geotransform

    @property
    def mask(self):
        """
        Returns a boolean mask where the raster contains
        no_data_value elements.
        """
        return self.data == self.no_data_value

    @property
    def origin(self):
        """
        Returns the lower left corner of the raster.
        """
        return (self.xll_corner, self.yll_corner)

    @origin.setter
    def origin(self, v):
        """
        Sets the lower-left corner of the raster.
        """
        self.xll_corner = v[0]
        self.yll_corner = v[1]

    @property
    def shape(self):
        return self.data.shape

    @property
    def nrows(self):
        return self.data.shape[0]

    @property
    def ncols(self):
        return self.data.shape[1]

    @property
    def extent(self):
        """
        Returns the spatial extent of the raster, in the form:

            ``(x_min, y_min, x_max, y_max)``
        """
        return (
            self.xll_corner,  # x_min
            self.yll_corner,  # y_min
            self.ncols * self.cell_size + self.xll_corner,  # x_max
            self.nrows * self.cell_size + self.yll_corner,  # y_max
        )

    @property
    def units(self):
        return self.crs.axis_info[0].unit_name

    @property
    def centroid(self):
        """
        Returns the raster centroid.
        """
        xmin, ymin, xmax, ymax = self.extent
        return (xmin + (xmax - xmin) / 2.0, ymin + (ymax - ymin) / 2.0)

    @property
    def max_cell(self):
        """Returns the first cell index representing the largest value in the raster."""
        masked_data = self.masked_data()
        idx = np.argwhere(masked_data == np.nanmax(masked_data))
        return tuple(idx[0])

    @property
    def min_cell(self):
        """Returns the first cell index representing the smallest value in the raster."""
        masked_data = self.masked_data()
        idx = np.argwhere(masked_data == np.nanmin(masked_data))
        return tuple(idx[0])

    @property
    def area(self):
        """
        Returns the surface area of the raster, ignoring `noDataValue` cells.
        """
        # Count the non-NaN cells in the raster.
        # Then, multiply by cell_size_x * cell_size_y to adjust for CRS.
        valid_cells = np.sum(~np.isnan(self.masked_data()))
        return (self.cell_size * self.cell_size) * valid_cells

    def values_at(self, points: np.ndarray):
        """
        Returns the raster values at `points`, where `points`
        is a coordinate X-Y array in the same CRS as the raster.
        """
        indices = unproject_vector(points, self)
        indices = (indices[:, 1], indices[:, 0])

        return self.masked_data()[indices]

    def value_at(self, x: float, y: float):
        """
        Returns the value of the raster at point (x, y) in the same CRS
        as the raster.
        """
        return self.values_at(np.array([[x, y]]))

    def masked_data(self):
        """
        Returns a copy of the raster data where all `no_data_value`
        elements in the raster are replaced with `numpy.nan`.
        """
        masked = np.array(deepcopy(self.data))
        masked[self.mask] = np.nan
        return masked

    def fill_depressions(self, fill_depressions: bool = True, fill_flats: bool = True):
        """
        Fills flats and depressions in a DEM raster.
        On meshes intended to be high-resolution, leaving flats and
        depressions untouched may cause solver issues. This method
        should be called before generating a triangle mesh from a
        DEM raster.

        # Arguments
        fill_depressions (bool): fill pits and depressions on DEM
        fill_flats (bool): fill flats on DEM
        """

        if fill_depressions:
            f = io.StringIO()

            with redirect_stdout(f):
                rd.FillDepressions(
                    self.data, in_place=True, epsilon=False, topology="D8"
                )

        if fill_flats:
            f = io.StringIO()

            with redirect_stdout(f):
                rd.ResolveFlats(self.data, in_place=True)

    def plot(
        self,
        layers: list = None,
        outfile: str = None,
        title: str = None,
        geometry: list = None,
        hillshade: bool = False,
    ):
        """
        Plots the raster object.

        # Arguments
        outfile (str): path to save figure
        title (str): figure title
        geometry (list): a list of geometrical objects to overlay
        """

        if title is None:
            title = (
                f'Raster: "{os.path.basename(self.filename)}" | CRS: "{self.crs.name}"'
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
            raster_hillshade=hillshade,
        )

    def get_boundary(self, distance: float = None) -> Geometry:
        """
        Get a line mesh with nodes seperated by distance `distance` that
        describe the boundary of this raster object.
        """

        connect_ends = True

        if distance is None:
            distance = self.cell_size * 2.0

        distance /= self.cell_size
        vertices, connectivity = st_boundary(
            self.masked_data(),
            np.nan,
            dist=distance,
            connect_ends=connect_ends,
        )

        vertices = project_vector(vertices, self)
        polygons = polygonize(linemerge(vertices[connectivity - 1].tolist()))

        return Geometry(
            shapes=list(polygons),
            crs=self.crs,
        )

    def save(self, outfile: str):
        """
        Saves a raster object to disk in GeoTIFF format. Writes array as Float64.
        """

        if extension(outfile) not in ["tif", "tiff"]:
            warn("Writing raster as a GeoTIFF.")

        debug(f"Attempting to save raster object to {outfile}")

        driver = gdal.GetDriverByName("GTiff")

        outdata = driver.Create(outfile, self.ncols, self.nrows, 1, gdal.GDT_Float64)
        outdata.SetGeoTransform(self.geotransform)
        outdata.SetProjection(
            self.crs.to_wkt(version=pyproj.enums.WktVersion.WKT1_GDAL)
        )
        outdata.GetRasterBand(1).WriteArray(np.array(self.data, dtype=np.float64))
        outdata.GetRasterBand(1).SetNoDataValue(self.no_data_value)
        outdata.FlushCache()

        log(f"Raster object saved to {outfile}")
