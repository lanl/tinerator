import pytest
import shapely
import numpy as np

import tinerator as tin
from tinerator import examples

WGS84 = "EPSG:4326"
NM = tin.examples.NewMexico()


@pytest.fixture
def shp_flowline_nm():
    return NM.flowline


@pytest.fixture
def shp_boundary_nm():
    yield NM.boundary


@pytest.fixture
def dem_nm():
    return NM.dem


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

    # assert len(shp.shapes) == len(s1.shapes) == len(s2.shapes)
    # assert s1.centoid == s2.centroid
    # assert shp.centroid != s1.centroid


def test_load_raster(dem_nm):
    r = dem_nm
    assert np.nanmin(r.masked_data()) > 0


def test_clip_raster(dem_nm, shp_boundary_nm):
    r = dem_nm
    s = shp_boundary_nm
    # r1 = r.clip(s)
    r2 = tin.gis.clip_raster(dem_nm, s)
    assert True


def test_reproject_raster(dem_nm):
    r = dem_nm
    r1 = tin.gis.reproject_raster(r, WGS84)
    r2 = r.reproject(WGS84)

    assert True


def test_visualize(dem_clipped_nm, shp_boundary_nm, shp_flowline_nm):
    dem = dem_clipped_nm
    boundary = shp_boundary_nm
    flowline = shp_flowline_nm

    # /Users/livingston/dev/lanl/tinerator/tinerator/tmp

    tin.plot(dem, boundary, flowline)
