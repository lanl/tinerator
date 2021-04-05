import richdem as rd
import numpy as np
import io
import os
import rasterio
import gdal
import pyproj
import tempfile
from contextlib import redirect_stdout
from copy import deepcopy
from pyproj import CRS
from pyproj.crs import CRSError
from ..logging import log, warn, debug, error
from ..visualize import plot as pl
from .utils import project_vector, unproject_vector, parse_crs
from .raster_boundary import square_trace_boundary as st_boundary
from .vector import Shape, ShapeType

# Rendering a DEM in 3D:
# https://pvgeo.org/examples/grids/read-esri.html#sphx-glr-examples-grids-read-esri-py

extension = lambda x: os.path.splitext(x)[-1].replace(".", "").lower().strip()


def load_raster(filename: str, no_data: float = None, to_crs: str = None):
    """
    Loads a raster from a given filename.
    """

    log(f"Loading raster from {filename}")

    r = Raster(filename, no_data=no_data)

    if to_crs is not None:
        r.reproject(to_crs)

    return r


def new_raster(
    data: np.ndarray, geotransform: tuple, crs: CRS, no_data: float = -3.0e16
):
    """
    Creates a new Raster object from a Numpy array.
    """

    data = np.array(data, dtype=np.float64)
    nrows, ncols = data.shape
    dtype = gdal.GDT_Float64

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
        arr = self.masked_data()

        # DEM information
        display = "\ntinerator.gis.Raster object\n"
        display += "=" * 50 + "\n"
        display += "Source\t\t: file (%s)\n" % self.filename
        display += "CRS\t\t: %s\n" % self.crs.name
        display += "Extent\t\t: %s\n" % repr(self.extent)
        display += "Cell size\t: %s\n" % repr(self.cell_size)
        display += "Units\t\t: %s\n" % self.units
        display += "Dimensions\t: %s\n" % repr((self.nrows, self.ncols))
        display += "NoDataValue\t: %s\n" % repr(self.no_data_value)
        display += "Value range\t: %s\n" % repr((np.nanmin(arr), np.nanmax(arr)))
        display += "\n%s\n" % repr(self.data)

        return display

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
        Returns the spatial extent of the raster.
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

    def get_boundary(self, distance: float = None, as_polygon: bool = False) -> Shape:
        """
        Get a line mesh with nodes seperated by distance `distance` that
        describe the boundary of this raster object.
        """

        if as_polygon:
            connect_ends = True
            shape_type = ShapeType.POLYGON
        else:
            connect_ends = False
            shape_type = ShapeType.POLYLINE

        if distance is None:
            distance = 10.0

        distance /= self.cell_size
        vertices, connectivity = st_boundary(
            self.masked_data(),
            np.nan,
            dist=distance,
            connect_ends=connect_ends,
        )

        return Shape(
            points=project_vector(vertices, self),
            crs=self.crs,
            shape_type=shape_type,
            connectivity=connectivity,
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
