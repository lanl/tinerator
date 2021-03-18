import os
import tinerator as tin
import numpy as np
import tempfile

TEST_SHAPEFILE = ""
TEST_DEM = ""

def test_load_raster():
    d1 = tin.load.load_raster(TEST_DEM)
    d2 = tin.load.load_raster(TEST_DEM, no_data=-9999.)

    assert d1 == d2

def test_load_shapefile():
    tin.load.load_shapefile(TEST_SHAPEFILE)
    assert True

def test_write_raster():
    with tempfile.TemporaryDirectory() as tmp_dir:
        TMP = os.path.join(tmp_dir, "raster.tif")
        d1 = tin.load.load_raster(TEST_DEM)
        d1.save(TMP)
        d2 = tin.load.load_raster(TMP)

        assert d1 == d2

def test_write_shapefile():
    pass

def test_clip_raster():
    dem = tin.gis.load_raster(sample)
    boundary = tin.gis.load_shapefile(sample2)
    tin.gis.clip_raster(dem, boundary)

def test_reproject_raster():
    pass

def test_reproject_shapefile():
    pass