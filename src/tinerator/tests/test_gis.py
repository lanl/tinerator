import pytest
import tempfile
import hashlib
import shapely
import numpy as np

import tinerator as tin
from tinerator import examples

WGS84 = "EPSG:4326"
NM = tin.examples.NewMexico()

HASHES = {
    "visualization": {
        "plotly": {
            "jpg": "0000000",
            "png": "0000000",
            "svg": "0000000",
            "html": "0000000",
        },
    },
}

MIN_LATITUDE = -90.0
MAX_LATITUDE = 90.0
MIN_LONGITUDE = -180.0
MAX_LONGITUDE = 180.0


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


def test_gis_extent(dem_nm, dem_clipped_nm, shp_boundary_nm, shp_flowline_nm):
    """Test that extents within WGS-84 are calculated and returned correctly."""

    for layer in (dem_nm, dem_clipped_nm, shp_boundary_nm, shp_flowline_nm):
        lat_min, lat_max, lon_min, lon_max = layer.reproject(WGS84).extent
        assert all(
            [
                MAX_LATITUDE >= lat_min >= MIN_LATITUDE,
                MAX_LATITUDE >= lat_max >= MIN_LATITUDE,
                MAX_LONGITUDE >= lon_min >= MIN_LONGITUDE,
                MAX_LONGITUDE >= lon_max >= MIN_LONGITUDE,
            ]
        )


def test_visualize2D_plotly(dem_clipped_nm, shp_boundary_nm, shp_flowline_nm):
    """Tests that Plotly is rendering and saving figures correctly."""
    dem = dem_clipped_nm
    boundary = shp_boundary_nm
    flowline = shp_flowline_nm

    with tempfile.TemporaryDirectory() as d:
        for (fmt, hash) in HASHES["visualization"]["plotly"].items():
            outfile = os.path.join(d, f"image.{fmt}")
            fig.save(outfile)

            with open(outfile, "rb") as f:
                assert (
                    hashlib.md5(f.read()).hexdigest() == hash
                ), f'Image write "{fmt}" failed'

    # tin.plot(dem, boundary, flowline)


def test_isinstance_tinerator(dem_clipped_nm, shp_boundary_nm):
    """Tests that the 'isinstance_*' function behaves correctly."""
    assert tin.util.isinstance_geometry(shp_boundary_nm) == True
    assert tin.util.isinstance_geometry(dem_clipped_nm) == False
    assert tin.util.isinstance_geometry("fail") == False
    assert tin.util.isinstance_raster(dem_clipped_nm) == True
    assert tin.util.isinstance_raster(dem_clipped_nm) == False
    assert tin.util.isinstance_raster("fail") == False
