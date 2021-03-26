import os
import sys
import numpy as np
import tinerator as tin
import meshio
from tinerator import ExampleData


def meshes_equal(test_fname, gold_fname) -> bool:
    mo_test = meshio.read(test_fname, file_format="avsucd")
    mo_gold = meshio.read(gold_fname, file_format="avsucd")

    # TODO: what if points are in different order?
    assert np.allclose(mo_test.points, mo_gold.points)
    assert len(mo_test.cells) == len(mo_gold.cells)

    # TODO: what if cells are in different order?
    for i in range(len(mo_gold.cells)):
        assert np.array_equal(mo_test.cells[i].data, mo_gold.cells[i].data)

    assert np.array_equal(
        list(mo_test.point_data.keys()), list(mo_gold.point_data.keys())
    )
    assert np.array_equal(
        list(mo_test.cell_data.keys()), list(mo_gold.cell_data.keys())
    )

    for key in mo_gold.point_data.keys():
        assert np.allclose(mo_test.point_data[key], mo_gold.point_data[key])

    for key in mo_gold.cell_data.keys():
        assert np.allclose(mo_test.cell_data[key], mo_gold.cell_data[key])

    return True


def test_raster_load():
    data = ExampleData.NewMexico
    _ = tin.gis.load_raster(data.dem)
    assert True


def test_meshing_workflow():
    data = ExampleData.Simple

    surface_mesh = tin.meshing.load_mesh(data.surface_mesh)
    tin.debug_mode()

    depths = [0.1, 0.3, 0.2, 0.1, 0.4]
    matids = [1, 2, 3, 3, 4]

    layers = tin.meshing.DEV_get_dummy_layers(surface_mesh, depths)

    volume_mesh = tin.meshing.DEV_stack(layers, matids=matids)
    volume_mesh.save("test_vol.inp")

    assert meshes_equal("test_vol.inp", data.volume_mesh)
