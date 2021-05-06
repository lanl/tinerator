import pyproj
import os
import fiona
import rasterio
import rasterio.mask
import shutil
import geopandas
import numpy as np
import tempfile
import snowy
from osgeo import ogr
from ..logging import log, warn, debug, error
from .raster import Raster, load_raster, new_raster
from .vector import Shape, ShapeType

try:
    from osgeo import gdal
except ImportError:
    import gdal

def get_geometry(shapefile_path: str) -> list:
    """
    Reads a shapefile and returns a list of dicts of
    all geometric objects within the shapefile. Each dict
    contains the type of geometrical object, its CRS, and
    defining coordinates.

    # Arguments
    shapefile_path (str): path to shapefile

    # Returns
    list[dict]
    """

    elements = []
    with fiona.open(shapefile_path, "r") as cc:
        for f in cc:
            geom = f["geometry"]
            coords = geom["coordinates"]

            if not isinstance(coords[0], tuple):
                coords = coords[0]

                if not isinstance(coords[0], tuple):
                    print("warning: shapefile parsed incorrectly")

            elements.append(
                {
                    "type": geom["type"],
                    "crs": None,
                    "coordinates": np.array(coords),
                }
            )

    return elements


def reproject_shapefile(
    shapefile_in: str, shapefile_out: str, crs: str = None, epsg: int = None
) -> None:
    """
    Transforms all geometries in a shapefile to a new CRS and writes
    to `shapefile_out`.

    Either `crs` or `epsg` must be specified. `crs` can be either a string or
    a dict.

    See `help(geopandas.geodataframe.GeoDataFrame.to_crs)` for more information.

    # Arguments
    shapefile_in (str): filepath to the shapefile
    shapefile_out (str): file to write re-projected shapefile to
    crs (str or dict): Proj4 string with new projection; i.e. '+init=epsg:3413'
    epsg (int): EPSG code specifying output projection
    """
    shp = geopandas.read_file(shapefile_in)
    shp = shp.to_crs(crs=crs, epsg=epsg)
    shp.to_file(shapefile_out, driver="ESRI Shapefile")


def reproject_raster_2(raster_in: str, raster_out: str, dst_crs: str) -> None:
    """
    Re-projects a raster and writes it to `raster_out`.

    # Example
    ```python
    reproject_raster('dem_in.asc','dem_out.tif','EPSG:2856')
    ```

    # Arguments
    raster_in (str): Filepath to input raster
    raster_out (str): Filepath to save reprojected raster
    dst_crs (str): Desired CRS
    """

    from rasterio.warp import (
        calculate_default_transform,
        reproject,
        Resampling,
    )

    with rasterio.open(raster_in) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update(
            {
                "crs": dst_crs,
                "transform": transform,
                "width": width,
                "height": height,
            }
        )

        with rasterio.open(raster_out, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest,
                )


def rasterize_shape(raster: Raster, shape: Shape) -> Raster:
    """
    Rasterizes a shape.
    """

    log(f"Rasterizing {shape} to {raster}")

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
    raster: Raster, shape: Shape, min_dist: float = 0.0, max_dist: float = 1.0
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

    log("Creating distance map")

    # Rasterize a shapefile onto the same dimensionality
    # and projection as `raster`
    shape_raster = rasterize_shape(raster, shape)

    data = np.array(shape_raster.data)
    data[data < 0] = 0
    data[data > 0] = 1
    data = data.astype(bool)

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

    log("Clipping raster with shapefile")

    if shape.shape_type != ShapeType.POLYGON:
        warn(
            f"Vector shape type must be polygon to clip raster, not {shape.shape_type}."
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


def reproject_raster(raster: Raster, dst_crs) -> Raster:

    with tempfile.TemporaryDirectory() as tmp_dir:
        debug(f"Temp directory created at: {tmp_dir}")

        RASTER_OUT = os.path.join(tmp_dir, "raster.tif")
        REPROJ_RASTER_OUT = os.path.join(tmp_dir, "raster_reprojected.tif")

        raster.save(RASTER_OUT)
        debug("Shapefile was saved from memory to disk")

        gdal.Warp(REPROJ_RASTER_OUT, RASTER_OUT, dstSRS=dst_crs, format="GTiff")

        return load_raster(REPROJ_RASTER_OUT)
