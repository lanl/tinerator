import os
import pkg_resources
from ..gis import load_raster, load_shapefile


# Gets absolute path relative to installed module location
def _get_data(rel_path: str) -> str:
    return pkg_resources.resource_filename('tinerator', f'examples/data/{rel_path}')


class NewMexico(object):
    """
    New Mexico GeoTIFF + shapefiles.
    """

    def __init__(self):
        self.dem = load_raster(_get_data('new_mexico/rasters/USGS_NED_13_n37w108_Clipped.tif'))
        self.flowline = load_shapefile(_get_data("new_mexico/shapefiles/NHDFlowline/NHDFlowline.shp"))
        self.boundary = load_shapefile(_get_data("new_mexico/shapefiles/WBDHU12/WBDHU12.shp"))


class Borden(object):
    """
    Borden simple test case.
    """

    def __init__(self):
        self.dem_50cm = load_raster(_get_data("borden/dem0.5m.dat"))
        self.dem_100cm = load_raster(_get_data("borden/dem1m.dat"))

    @property
    def dem(self):
        # Helper property to minimize confusion over
        # the multiple DEMs present
        return self.dem_50cm
