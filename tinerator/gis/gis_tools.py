import gdal
import os
import fiona
import rasterio
import rasterio.mask
import shutil
import geopandas
import numpy as np
import tempfile
from ..logging import log, warn, debug, error
from .raster import load_raster

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

def clip_raster(raster, shapefile):
    log(f"Clipping raster with shapefile")

    with tempfile.TemporaryDirectory() as tmp_dir:
        debug(f"Temp directory created at: {tmp_dir}")

        RASTER_OUT = os.path.join(tmp_dir, "raster.tif")
        VECTOR_OUT = os.path.join(tmp_dir, "shape.shp")
        CLIPPED_RASTER_OUT = os.path.join(tmp_dir, "raster_clipped.tif")

        raster.save(RASTER_OUT)
        shapefile.save(VECTOR_OUT)

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
