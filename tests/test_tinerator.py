import os
import sys
import numpy as np
import tinerator as tin
import meshio
from tinerator import ExampleData
from tempfile import TemporaryDirectory

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


# def test_get_boundary():
#    boundary = dem.get_boundary(distance=1)

# def test_clip_raster():
#    boundary = tin.gis.load_shapefile
#    dem2 = tin.gis.clip_raster(dem, boundary)
#    assert something

# def test_clip_raster_with_boundary():
#     boundary = dem.get_boundary(distance=1)
#     dem2 = tin.gis.clip_raster(dem, boundary)
#     assert something

# def test_reproject():
#    usgs_dem = tin.gis.reproject_raster(usgs_dem, 'EPSG:32112')
#    usgs_dem = tin.gis.reproject_shapefile(usgs_dem, 'EPSG:32112')#
#    dem.reproject('EPSG:6666')
#    shapefile.reproject('EPSG:234')

def test_raster_load():
    data = ExampleData.NewMexico
    _ = tin.gis.load_raster(data.dem)
    assert True


def test_triangulate():
    return True
    data = ExampleData.NewMexico

    dem = tin.gis.load_raster(data.dem)
    boundary = tin.gis.load_shapefile(data.watershed_boundary)
    dem = tin.gis.clip_raster(dem, boundary)
    dem = tin.gis.reproject_raster(dem, "EPSG:32112")

    # TODO: 
    # Be able to handle geometries with multiple shapes
    # Also, this is a: shapefile.POLYLINEZ
    # Throws error
    # flowline = tin.gis.load_shapefile(data.flowline)

    flowline = tin.gis.load_shapefile("/Users/livingston/playground/lanl/tinerator/tmp/shapeflriver.shp")

    for method in ["meshpy", "jigsaw"]:
        surf = tin.meshing.triangulate(
            dem, 
            min_edge_length=0.01, 
            max_edge_length = 0.1, 
            scaling_type="relative", 
            method=method,
            refinement_feature=flowline,
        )
        surf.save("test_surf.vtk")


def test_meshing_workflow():
    return True
    data = ExampleData.Simple

    surface_mesh = tin.meshing.load_mesh(data.surface_mesh)
    tin.debug_mode()

    depths = [0.1, 0.3, 0.2, 0.1, 0.4]
    matids = [1, 2, 3, 3, 4]

    layers = tin.meshing.DEV_get_dummy_layers(surface_mesh, depths)

    volume_mesh = tin.meshing.DEV_stack(layers, matids=matids)
    volume_mesh.save("test_vol.inp")
    assert meshes_equal("test_vol.inp", data.volume_mesh)

    tin.meshing.DEV_spit_out_simple_mesh(volume_mesh)  # DEV_basic_facesets(volume_mesh)

def test_exodus_write():
    example = tin.ExampleData.Simple
    volume_mesh = tin.meshing.load_mesh(example.volume_mesh)

    tin.debug_mode()
    surf_mesh = tin.meshing.SurfaceMesh(volume_mesh)
    top_faces = surf_mesh.top_faces
    bottom_faces = surf_mesh.bottom_faces
    side_faces = surf_mesh.side_faces
    top_points = surf_mesh.top_points

    with TemporaryDirectory() as tmp_dir:
        tmp_dir = "/Users/livingston/playground/lanl/tinerator/tmp/exo_sets"
        surf_mesh.save(os.path.join(tmp_dir, "surf.inp"))
        outfile = os.path.join(tmp_dir, "mesh_out.exo")

        import ipdb; ipdb.set_trace()

        tin.meshing.dump_exodus(
            outfile,
            volume_mesh.nodes,
            volume_mesh.elements,
            cell_block_ids=volume_mesh.material_id,
            side_sets=[top_faces, bottom_faces, side_faces],
            node_sets=[top_points],
        )

        diff = tin.meshing.check_mesh_diff(outfile, example.exodus_mesh)

        assert len(diff) < 50

'''
<   "CMO_NAME",
---
>   "mo3",
167,169c167,169
TIN
<  elem_ss3 = 1, 1, 3, 4, 5, 6, 6, 7, 9, 9, 11, 12, 13, 14, 14, 15, 17, 17, 19, 
<     20, 21, 22, 22, 23, 25, 25, 27, 28, 29, 30, 30, 31, 33, 33, 35, 36, 37, 
<     38, 38, 39 ;
---
LG
>  elem_ss3 = 1, 1, 2, 4, 5, 7, 8, 8, 9, 9, 10, 12, 13, 15, 16, 16, 17, 17, 18, 
>     20, 21, 23, 24, 24, 25, 25, 26, 28, 29, 31, 32, 32, 33, 33, 34, 36, 37, 
>     39, 40, 40 ;
171,172c171,172
<  side_ss3 = 2, 3, 2, 1, 2, 3, 1, 3, 2, 3, 2, 1, 2, 3, 1, 3, 2, 3, 2, 1, 2, 3, 
<     1, 3, 2, 3, 2, 1, 2, 3, 1, 3, 2, 3, 2, 1, 2, 3, 1, 3 ;
---
>  side_ss3 = 1, 3, 1, 2, 3, 2, 2, 3, 1, 3, 1, 2, 3, 2, 2, 3, 1, 3, 1, 2, 3, 2, 
>     2, 3, 1, 3, 1, 2, 3, 2, 2, 3, 1, 3, 1, 2, 3, 2, 2, 3 ;
'''
#np.sort(m1[mapping]-1)