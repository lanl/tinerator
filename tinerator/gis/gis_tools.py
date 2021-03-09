import gdal
from osgeo import ogr
import pyproj
import os
import fiona
import rasterio
import rasterio.mask
import shutil
import geopandas
import numpy as np
import tempfile
from ..logging import log, warn, debug, error
from .raster import Raster, load_raster, new_raster
from .vector import Shape, ShapeType

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


def reproject_raster(raster_in: str, raster_out: str, dst_crs: str) -> None:
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
    '''
    Rasterizes a shape.
    '''

    log(f"Rasterizing {shape} to {raster}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = "/Users/livingston/playground/lanl/tinerator/tinerator-test-cases/tmp/huh"
        debug(f"Temp directory created at: {tmp_dir}")

        RASTER_OUT = os.path.join(tmp_dir, "raster.tif")
        VECTOR_OUT = os.path.join(tmp_dir, "shape.shp")
        RASTERIZE_OUT = os.path.join(tmp_dir, "rasterized_shape.tif")

        raster.save(RASTER_OUT)
        shape.save(VECTOR_OUT)

        VECTOR_OUT = "/Users/livingston/Downloads/tmp/test.shp"

        # Read the shapefile with OGR
        shp_fh = ogr.Open(VECTOR_OUT)
        shp_layer = shp_fh.GetLayer()

        driver = gdal.GetDriverByName("GTiff")
        outdata = driver.Create(RASTERIZE_OUT, raster.ncols, raster.nrows, 1, gdal.GDT_Float64)
        outdata.SetGeoTransform(raster.geotransform)
        outdata.SetProjection(raster.crs.to_wkt(version=pyproj.enums.WktVersion.WKT1_GDAL))
        outdata.GetRasterBand(1).SetNoDataValue(raster.no_data_value)
        outdata.GetRasterBand(1).FlushCache()
        gdal.RasterizeLayer(outdata, [1], shp_layer, burn_values=[255])#, options=["ATTRIBUTE=hedgerow"])
        outdata.FlushCache()

        # Deleting this forces GDAL to flush to disk
        del outdata

        return load_raster(RASTERIZE_OUT)

def distance_map(raster: Raster, shape: Shape) -> Raster:
    '''
    Creates a distance map.
    '''

    # Refernces: 
    # Marching Parabolas algorithm
    # https://prideout.net/blog/distance_fields/
    # http://cs.brown.edu/people/pfelzens/dt/

    log("Creating distance map")

    shape_raster = rasterize_shape(raster, shape)

    nrows, ncols = shape_raster.shape

    #x = np.repeat(list(range(ncols)), nrows)
    #y = np.repeat(list(range(nrows)), ncols)

    print(nrows, ncols)

    x, y = np.meshgrid(np.linspace(1,ncols,80), np.linspace(1,nrows,40)) 
    dst = np.sqrt(x*x+y*y) 
    
    # Intializing sigma and muu 
    sigma = 100
    muu = +300.000
    
    # Calculating Gaussian array 
    gauss = np.exp(-( (dst-muu)**2 / ( 2.0 * sigma**2 ) ) ) 

    #all_nodes = np.array([x,y]).T
    #shape_nodes = np.argwhere(shape_raster.data == 255)

    #debug(f"Attempting distance map between {shape_nodes.shape} and {all_nodes.shape} nodes")

    #distance_map = new_raster(gauss)
    return gauss


def clip_raster(raster: Raster, shape: Shape) -> Raster:
    '''
    Returns a new Raster object, clipped by a Shape polygon.

    dem = tin.gis.load_raster("raster.tif")
    boundary = tin.gis.load_shapefile("boundary.shp")
    new_dem = tin.gis.clip_raster(dem, boundary)
    '''

    log(f"Clipping raster with shapefile")

    if shape.shape_type != ShapeType.POLYGON:
        warn(f"Vector shape type must be polygon to clip raster, not {shape.shape_type}.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        debug(f"Temp directory created at: {tmp_dir}")

        RASTER_OUT = os.path.join(tmp_dir, "raster.tif")
        VECTOR_OUT = os.path.join(tmp_dir, "shape.shp")
        CLIPPED_RASTER_OUT = os.path.join(tmp_dir, "raster_clipped.tif")

        raster.save(RASTER_OUT)
        shape.save(VECTOR_OUT)

        debug(f"Shapefile was saved from memory to disk")

        gdal.Warp(
            CLIPPED_RASTER_OUT, 
            RASTER_OUT, 
            format = 'GTiff',
            outputType = gdal.GDT_Float64, 
            cutlineDSName = VECTOR_OUT,
            cropToCutline = True
        )

        return load_raster(CLIPPED_RASTER_OUT)
