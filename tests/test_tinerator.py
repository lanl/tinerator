import os
import sys
import numpy as np
import tinerator as tin
import meshio
from tinerator import ExampleData
from util import meshes_equal
from tempfile import TemporaryDirectory

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

    with TemporaryDirectory() as tmp_dir:
        outfile = os.path.join(tmp_dir, "mesh_out.exo")

        tin.meshing.dump_exodus(
            outfile,
            volume_mesh.nodes,
            volume_mesh.elements,
            cell_block_ids=volume_mesh.material_id,
        )

        diff = tin.meshing.check_mesh_diff(outfile, example.exodus_mesh)

        assert len(diff) < 50