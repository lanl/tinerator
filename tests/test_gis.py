import pytest
import shapely
import numpy as np

import tinerator as tin
from tinerator import examples

WGS84 = 'EPSG:4326'

@pytest.fixture
def shp_flowline_nm():
    return tin.gis.load_shapefile(examples.new_mexico.flowline)

@pytest.fixture
def shp_boundary_nm():
    yield tin.gis.load_shapefile(examples.new_mexico.boundary)

@pytest.fixture
def dem_nm():
    return tin.gis.load_raster(examples.new_mexico.dem)

@pytest.fixture
def dem_clipped_nm():
    return tin.gis.clip_raster(dem_nm, shp_boundary_nm)

# ==================================== #

def test_load_shapefiles(shp_flowline_nm):
    shp = shp_flowline_nm
    assert len(shp.shapes) > 0

def test_reproject_shapefile(shp_boundary_nm):
    shp = shp_boundary_nm

    s1 = shp.reproject(WGS84)
    s2 = tin.gis.reproject_geometry(shp, WGS84)

    assert True

    #assert len(shp.shapes) == len(s1.shapes) == len(s2.shapes)
    #assert s1.centoid == s2.centroid
    #assert shp.centroid != s1.centroid

def test_load_raster(dem_nm):
    r = dem_nm
    assert np.nanmin(r.masked_data()) > 0

def test_clip_raster(dem_nm, shp_boundary_nm):
    r = dem_nm
    s = shp_boundary_nm
    #r1 = r.clip(s)
    r2 = tin.gis.clip_raster(dem_nm, s)
    assert True

def test_reproject_raster(dem_nm):
    r = dem_nm
    r1 = tin.gis.reproject_raster(r, WGS84)
    r2 = r.reproject(WGS84)

    assert True