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

    tin.meshing.DEV_basic_facesets(volume_mesh)


"""
    _cleanup.append(boundary_file)
    dem_object._stacked_mesh.resetpts_itp()

    dem_object.lg.sendline('resetpts/itp')
    dem_object.lg.sendline('createpts/median')
    dem_object.lg.sendline('sort/{0}/index/ascending/ikey/itetclr zmed ymed xmed'.format(dem_object._stacked_mesh.name))
    dem_object.lg.sendline('reorder/{0}/ikey'.format(dem_object._stacked_mesh.name))
    dem_object.lg.sendline('cmo/DELATT/{0}/ikey'.format(dem_object._stacked_mesh.name))
    dem_object.lg.sendline('cmo/DELATT/{0}/xmed'.format(dem_object._stacked_mesh.name))
    dem_object.lg.sendline('cmo/DELATT/{0}/ymed'.format(dem_object._stacked_mesh.name))
    dem_object.lg.sendline('cmo/DELATT/{0}/zmed'.format(dem_object._stacked_mesh.name))
    dem_object.lg.sendline('cmo/DELATT/{0}/ikey'.format(dem_object._stacked_mesh.name))

    cmo_in = dem_object._stacked_mesh.copy()

    # Extract surface w/ cell & face attributes to get the outside face
    # to element relationships

    try:
        raise Exception('Unknown bug in standard surfmesh...move to catch')
        mo_surf = dem_object.lg.extract_surfmesh(cmo_in=cmo_in,
                                                 stride=[1,0,0],
                                                 external=True,
                                                 resetpts_itp=True)
"""
