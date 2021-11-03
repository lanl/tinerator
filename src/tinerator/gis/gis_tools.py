import pyproj
import os
import numpy as np
import tempfile
import snowy
from shapely.ops import transform
from .geoutils import parse_crs
from ..logging import log, warn, debug, error
from .raster import Raster, load_raster, new_raster
from .vector import Shape, ShapeType
from .geometry import Geometry
from typing import Any, Callable, List, Union

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


def reproject_geometry(
    shape: Geometry, crs: Union[pyproj.CRS, str, int, dict]
) -> Geometry:
    """
    Reprojects a Geometry object into the provided CRS and returns
    the new object.

    Args:
        shape (Geometry): The Geometry object to reproject.
        crs (Union[pyproj.CRS, str, int, dict]): The CRS to reproject into.

    Returns:
        A TINerator Geometry object in the new coordinate reference space.
    """

    crs = parse_crs(crs)
    project = pyproj.Transformer.from_crs(shape.crs, crs)
    new_shapes = [transform(project.transform, shp) for shp in shape.shapes]

    return Geometry(shapes=new_shapes, crs=crs, properties=shape.properties)


def rasterize_geometry(raster: Raster, shape: Shape) -> Raster:
    """
    Rasterizes a Geometry object into a raster similar to :obj:`raster`.

    Args:
        raster (Raster): The TINerator Raster object to use as a template
            for rasterizing the shape.
        shape (Geometry): The TINerator Geometry object to rasterize.

    Returns:
        A TINerator Raster object with the Geometry burned into a
        raster of the same extent, projection, and size as :obj:`raster`.

    Examples:
        >>> rasterized_shape = tin.gis.rasterize_geometry(dem, watershed_flowline)
    """

    debug(f"Rasterizing {shape} to {raster}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        debug(f"Temp directory created at: {tmp_dir}")

        RASTER_OUT = os.path.join(tmp_dir, "raster.tif")
        VECTOR_OUT = os.path.join(tmp_dir, "shape.shp")
        RASTERIZE_OUT = os.path.join(tmp_dir, "rasterized_shape.tif")

        raster.save(RASTER_OUT)
        shape.save(VECTOR_OUT)

        # Read the shapefile with OGR
        shp_fh = ogr.Open(VECTOR_OUT)
        shp_layer = shp_fh.GetLayer()

        driver = gdal.GetDriverByName("GTiff")
        outdata = driver.Create(
            RASTERIZE_OUT, raster.ncols, raster.nrows, 1, gdal.GDT_Float64
        )
        outdata.SetGeoTransform(raster.geotransform)
        outdata.SetProjection(
            raster.crs.to_wkt(version=pyproj.enums.WktVersion.WKT1_GDAL)
        )
        outdata.GetRasterBand(1).SetNoDataValue(raster.no_data_value)
        outdata.GetRasterBand(1).FlushCache()
        gdal.RasterizeLayer(
            outdata, [1], shp_layer, burn_values=[255]
        )  # , options=["ATTRIBUTE=hedgerow"])
        outdata.FlushCache()

        # Deleting this forces GDAL to flush to disk
        del outdata

        return load_raster(RASTERIZE_OUT)


def distance_map(
    raster: Raster, shapes: Geometry, min_dist: float = 0.0, max_dist: float = 1.0
) -> Raster:
    """
    Creates a distance map. A new raster will be returned, where
    every cell will contain the (normalized) distance to the nearest
    intersection on :obj:`shape`.

    By default, the values will be normalized from 0 to 1.
    Setting `min_dist` and `max_dist` adjust that range.

    Args:
        raster (tinerator.gis.Raster): The input raster.
        shape (tinerator.gis.Shape): The shape to measure distance from.
        min_dist (:obj:`float`, optional): Defaults to 0.
        max_dist (:obj:`float`, optional): Defaults to 1.

    Returns:
        A raster filled with the normalized distance from each cell
        to the nearest intersection on `shape`.

    Note:
        It is rare that a user would want to call this. This is
        mainly an internal function used for triangulation.

    Examples:
        >>> dist = tin.gis.distance_map(dem, flowline)
    """

    debug("Creating distance map")

    # Rasterize a shapefile onto the same dimensionality
    # and projection as `raster`
    # If more than one shapefile, use bitwise-or to join together
    if not isinstance(shapes, list):
        shapes = [shapes]

    data = None

    for shape in shapes:
        shape_raster = rasterize_geometry(raster, shape)

        tmp_data = np.array(shape_raster.data)
        tmp_data[tmp_data < 0] = 0
        tmp_data[tmp_data > 0] = 1
        tmp_data = tmp_data.astype(bool)

        if data is None:
            data = tmp_data
        else:
            data = data | tmp_data

    # Use Snowy to generate a signed distance field
    # (The weird "[:,:,None]" slicing is due to Snowy
    #  expecting a "channels" dimension (i.e. for RGB))
    dmap = snowy.generate_sdf(data[:, :, None])[:, :, 0]

    # Turn it from a signed distance field to unsigned
    dmap[dmap < 0.0] = 0.0

    # Normalize distance values to [0, 1]
    dmap /= np.nanmax(dmap)

    # Now, reproject dfield value range to [min_dist, max_dist]
    dmap = dmap * (max_dist - min_dist) + min_dist

    return new_raster(
        data=dmap,
        geotransform=shape_raster.geotransform,
        crs=shape_raster.crs,
        no_data=-9999.0,
    )


def clip_raster(raster: Raster, shape: Shape) -> Raster:
    """
    Returns a new Raster object, clipped by a Shape polygon.
    The raster will have the same shape and projection as the
    previous one, but cells not covered by :obj:`shape` will
    be filled with the no data value of the parent raster.

    Args:
        raster (tinerator.gis.Raster): The raster to clip.
        shape (tinerator.gis.Shape): The shape to clip the raster with.

    Returns:
        A clipped raster.

    Examples:
        >>> dem = tin.gis.load_raster("raster.tif")
        >>> boundary = tin.gis.load_shapefile("boundary.shp")
        >>> new_dem = tin.gis.clip_raster(dem, boundary)
    """

    debug("Clipping raster with shapefile")

    if shape.geometry_type not in [
        "Polygon",
        "MultiPolygon",
        "3D Polygon",
        "3D MultiPolygon",
    ]:
        raise ValueError(
            f"Vector shape type must be polygon to clip raster, not {shape.geometry_type}."
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        debug(f"Temp directory created at: {tmp_dir}")

        RASTER_OUT = os.path.join(tmp_dir, "raster.tif")
        VECTOR_OUT = os.path.join(tmp_dir, "shape.shp")
        CLIPPED_RASTER_OUT = os.path.join(tmp_dir, "raster_clipped.tif")

        raster.save(RASTER_OUT)
        shape.save(VECTOR_OUT)

        debug("Shapefile was saved from memory to disk")

        gdal.Warp(
            CLIPPED_RASTER_OUT,
            RASTER_OUT,
            format="GTiff",
            outputType=gdal.GDT_Float64,
            cutlineDSName=VECTOR_OUT,
            cropToCutline=True,
        )

        return load_raster(CLIPPED_RASTER_OUT)


def reproject_raster(raster: Raster, crs: Union[pyproj.CRS, str, dict, int]) -> Raster:
    """
    Reprojects a TINerator Raster object into the
    destination CRS.

    The CRS must be a ``pyproj.CRS`` object, a WKT string,
    an EPSG code (in the style of "EPSG:1234"), or a
    PyProj string.

    Args:
        raster (Raster): A TINerator Raster object to reproject.
        dst_crs (pyproj.CRS): The CRS to project into.

    Returns:
        A reprojected Raster object.

    Examples:
        >>> tin.gis.reproject_raster(dem, "EPSG:3857")
    """

    dst_crs = parse_crs(crs)
    debug(f"Reprojecting from {raster.crs.name} into {dst_crs.name}")

    dst_crs = dst_crs.to_wkt(version=pyproj.enums.WktVersion.WKT1_GDAL)
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromWkt(dst_crs)

    debug(f"dst_crs = {dst_crs}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        debug(f"Temp directory created at: {tmp_dir}")

        RASTER_OUT = os.path.join(tmp_dir, "raster.tif")
        REPROJ_RASTER_OUT = os.path.join(tmp_dir, "raster_reprojected.tif")

        raster.save(RASTER_OUT)
        debug("Raster was saved from memory to disk")

        gdal.Warp(REPROJ_RASTER_OUT, RASTER_OUT, dstSRS=dst_srs, format="GTiff")

        return load_raster(REPROJ_RASTER_OUT)


def resample_raster(
    raster: Raster,
    new_res: tuple = None,
    new_shape: tuple = None,
    resampling_method: str = "near",
) -> Raster:
    """
    Resamples a raster into either/both of:
        * A new resolution in geospatial units, as (xRes, yRes)
        * A new raster shape, in (rows, cols)

    Various resampling algorithms are available on the GDAL website.

    https://gdal.org/programs/gdalwarp.html#cmdoption-gdalwarp-r

    Args:
        raster (Raster): A TINerator Raster object.
        new_res (:obj:`Tuple[float, float]`, optional): The new resolution in geospatial coordinates.
        new_shape (:obj:`Tuple[int, int]`, optional): The new number of rows and columns for the raster.

    Returns:
        A TINerator Raster object

    Examples:
        >>> new_shape = (x//2 for x in dem.shape)
        >>> new_dem = tin.gis.resample_raster(dem, new_shape = new_shape)
    """

    if new_res is not None:
        xRes, yRes = new_res
    else:
        xRes = yRes = None

    if new_shape is not None:
        rows, cols = new_shape
    else:
        rows, cols = (0, 0)

    with tempfile.TemporaryDirectory() as tmp_dir:
        RASTER_OUT = os.path.join(tmp_dir, "raster.tif")
        RESAMPLED_RASTER_OUT = os.path.join(tmp_dir, "raster_resampled.tif")

        raster.save(RASTER_OUT)

        args = gdal.WarpOptions(
            xRes=xRes, yRes=yRes, width=cols, height=rows, resampleAlg=resampling_method
        )

        gdal.Warp(RESAMPLED_RASTER_OUT, RASTER_OUT, options=args)

        return load_raster(RESAMPLED_RASTER_OUT)
