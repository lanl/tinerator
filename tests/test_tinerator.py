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

def test_get_boundary():
    data = ExampleData.NewMexico
    dem = tin.gis.load_shapefile(data.dem)
    boundary = dem.get_boundary(distance=1)
    assert True

def test_clip_raster():
   data = ExampleData.NewMexico
   dem = tin.gis.load_raster(data.dem)
   boundary = tin.gis.load_shapefile(data.watershed_boundary)
   _ = tin.gis.clip_raster(dem, boundary)
   assert True

def test_clip_raster_with_boundary():
    data = ExampleData.NewMexico
    dem = tin.gis.load_raster(data.dem)
    boundary = dem.get_boundary(distance=1)
    _ = tin.gis.clip_raster(dem, boundary)
    assert True

def test_reproject():
    data = ExampleData.NewMexico
    dem = tin.gis.load_raster(data.dem)
    boundary = tin.gis.load_shapefile(data.watershed_boundary)

    _ = tin.gis.reproject_raster(dem, 'EPSG:32112')
    _ = tin.gis.reproject_shapefile(boundary, 'EPSG:32112')
    _ = dem.reproject('EPSG:32112')
    _ = boundary.reproject('EPSG:32112')

def test_raster_load():
    data = ExampleData.NewMexico
    _ = tin.gis.load_raster(data.dem)
    assert True

def test_triangulate():
    data = ExampleData.NewMexico

    dem = tin.gis.load_raster(data.dem)
    boundary = tin.gis.load_shapefile(data.watershed_boundary)
    flowline = tin.gis.load_shapefile(data.flowline)

    dem = tin.gis.clip_raster(dem, boundary)
    dem = tin.gis.reproject_raster(dem, "EPSG:32112")
    
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
        surf_mesh.save(os.path.join(tmp_dir, "surf.inp"))
        outfile = os.path.join(tmp_dir, "mesh_out.exo")

        tin.meshing.dump_exodus(
            outfile,
            volume_mesh.nodes,
            volume_mesh.elements,
            cell_block_ids=volume_mesh.material_id,
            side_sets=[top_faces, bottom_faces, side_faces],
            node_sets=[top_points],
        )

        diff = tin.meshing.check_mesh_diff(outfile, example.exodus_mesh)

        assert len(diff) < 66
