import pytest
import shapely
import numpy as np
import tinerator as tin
data = tin.ExampleData

@pytest.fixture
def shp_flowline():
    return tin.gis.load_shapefile(data.NewMexico.flowline)

@pytest.fixture
def shp_boundary():
    return tin.gis.load_shapefile(data.NewMexico.watershed_boundary)

@pytest.fixture
def dem():
    r = tin.gis.load_raster(data.NewMexico.dem)
    return tin.gis.clip_raster(r, shp_boundary)

def test_meshpy():
    tri = tin.meshing.triangulate(
        dem,
        min_edge_length=0.01,
        max_edge_length=0.10,
        refinement_feature=shp_flowline,
        method="meshpy"
    )