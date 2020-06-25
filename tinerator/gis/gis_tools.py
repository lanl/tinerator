import os
import fiona
import rasterio
import rasterio.mask
import shutil
import geopandas


def reproject_shapefile(
    shapefile_in: str, shapefile_out: str, projection: str
) -> None:
    """
    Re-projects a shapefile and writes it to `shapefile_out`.

    # Arguments
    shapefile_in (str): filepath to the shapefile
    shapefile_out (str): file to write re-projected shapefile to
    projection (str): Proj4 string with new projection; i.e. '+init=epsg:3413'

    """
    shp = geopandas.read_file(shapefile_in)
    shp = shp.to_crs(projection)
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


def mask_raster(
    raster_filename: str, shapefile_filename: str, raster_outfile: str
):
    """
    Reads a raster file and ESRI shapefile and writes out
    a new raster cropped by the shapefile. 

    Note: both the raster and shapefile must be in the same
    CRS.

    # Arguments
    raster_filename (str): Raster file to be cropped
    shapefile_filename (str): Shapefile to crop raster with
    raster_outfile (str): Filepath to save cropped raster
    """

    # Capture the shapefile geometry
    with fiona.open(shapefile_filename, "r") as _shapefile:
        _poly = [feature["geometry"] for feature in _shapefile]

    # Open the DEM && mask && update metadata with mask
    with rasterio.open(raster_filename, "r") as _dem:
        out_image, out_transform = rasterio.mask.mask(
            _dem, _poly, crop=True, invert=False
        )
        out_meta = _dem.meta.copy()

    # Update raster metadata with new changes
    out_meta.update(
        {
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
        }
    )

    # TODO: add check that both are in same projection

    # Write out masked raster
    with rasterio.open(raster_outfile, "w", **out_meta) as dest:
        dest.write(out_image)
