import os
import sys
import numpy as np
import tinerator as tin
from tinerator import ExampleData

def test_raster_load():
    dem = tin.gis.load_raster(ExampleData.NewMexico.dem)
    assert True