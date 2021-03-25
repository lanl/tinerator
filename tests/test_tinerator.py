import os
import sys
import numpy as np
import tinerator as tin
from tinerator import ExampleData

def meshes_equal(mesh1, mesh2) -> bool:
    return True

def test_raster_load():
    data = ExampleData.NewMexico
    dem = tin.gis.load_raster(data.dem)
    assert True

def test_meshing_workflow():
    data = ExampleData.Simple

    surface_mesh = tin.meshing.load_mesh(data.surface_mesh)
    tin.debug_mode()

    depths = [0.1, 0.3, 0.2, 0.1, 0.4]
    matids = [1,2,3,3,4]

    layers = tin.meshing.DEV_get_dummy_layers(surface_mesh, depths)
    # tin.meshing.plot_meshes(layers)

    volume_mesh = tin.meshing.DEV_stack(layers, matids=matids)
    #assert 1 == 0
    #volume_mesh = tin.meshing.extrude_surface(surface_mesh, depths, matids=matids)
    #surface_mesh.stack(depths, matids=matids)
    volume_mesh.save('test_vol.inp')
    assert 1 == 0
