import pytest
import shapely
import numpy as np

import tinerator as tin
from tinerator import examples

@pytest.fixture(scope="module")
def shp_flowline():
    yield tin.gis.load_shapefile(examples.new_mexico.flowline)

@pytest.fixture(scope="module")
def shp_boundary():
    yield tin.gis.load_shapefile(examples.new_mexico.boundary)

@pytest.fixture(scope="module")
def dem(shp_boundary):
    r = tin.gis.load_raster(examples.new_mexico.dem)
    yield tin.gis.clip_raster(r, shp_boundary)

def test_meshpy_unrefined(dem, shp_flowline):
    tri = tin.meshing.triangulate(
        dem,
        min_edge_length=0.03,
        max_edge_length=0.10,
        scaling_type="relative",
        method="meshpy"
    )

    assert True

def test_meshpy_refined(dem, shp_flowline):
    tri = tin.meshing.triangulate(
        dem,
        min_edge_length=0.03,
        max_edge_length=0.10,
        scaling_type="relative",
        refinement_feature=shp_flowline,
        method="meshpy"
    )

    assert True

def test_jigsaw_unrefined(dem, shp_flowline):
    tri = tin.meshing.triangulate(
        dem,
        min_edge_length=0.03,
        max_edge_length=0.10,
        scaling_type="relative",
        method="jigsaw"
    )

    assert True

def test_jigsaw_refined(dem, shp_flowline):
    tri = tin.meshing.triangulate(
        dem,
        min_edge_length=0.03,
        max_edge_length=0.10,
        scaling_type="relative",
        refinement_feature=shp_flowline,
        method="jigsaw"
    )

    assert True