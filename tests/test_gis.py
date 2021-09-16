import pytest
import shapely
import numpy as np
import tinerator as tin
data = tin.ExampleData

@pytest.fixture
def shp_flowline_nm():
    return tin.gis.load_shapefile(data.NewMexico.flowline)

@pytest.fixture
def shp_boundary_nm():
    return tin.gis.load_shapefile(data.NewMexico.watershed_boundary)

@pytest.fixture
def dem_nm():
    return tin.gis.load_raster(data.NewMexico.dem)

@pytest.fixture
def dem_clipped_nm():
    return tin.gis.clip_raster(dem_nm, shp_boundary_nm)

# ==================================== #

def test_load_shapefiles():
    shp = shp_flowline_nm
    assert len(shp.shapes) > 0

def test_reproject_shapefile():
    TARGET_PROJ = "EPSG:1234"
    shp = shp_boundary_nm

    s1 = shp.reproject(TARGET_PROJ)
    s2 = tin.gis.reproject_geometry(shp)

    assert len(shp.shapes) == len(s1.shapes) == len(s2.shapes)
    assert s1.centoid == s2.centroid
    assert shp.centroid != s1.centroid

def test_load_raster():
    r = dem_nm
    assert np.nanmin(r.masked_data()) > 0

def test_clip_raster():
    r = dem_nm
    s = shp_boundary_nm
    r1 = r.clip(s)
    r2 = tin.gis.clip_raster(s)

    assert r1 == r2
    assert r.centroid == r1.centroid == r2.centroid
    pass

def test_reproject_raster():
    pass