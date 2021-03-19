import os
import sys
import numpy as np
import tinerator as tin

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "example-data")

class ExampleData:
    class NewMexico:
        root_dir = os.path.join(DATA_DIR, "GIS", "NewMexico-HU8-14080103")
        dem = os.path.join(DATA_DIR, "GIS", "NewMexico-HU8-14080103", "rasters", "USGS_NED_13_n37w108_Clipped.tif")
        watershed_boundary = os.path.join(DATA_DIR, "GIS", "NewMexico-HU8-14080103", "shapefiles", "WBDHU12", "WBDHU12.shp")
        flowline = os.path.join(DATA_DIR, "GIS", "NewMexico-HU8-14080103", "shapefiles", "NHDFlowline", "NHDFlowline.shp")

def test_raster_load():
    dem = tin.gis.load_raster(ExampleData.NewMexico.dem)
    assert True