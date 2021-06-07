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


def test_geometry_read():
    data = ExampleData.NewMexico
    shp = tin.gis.load_shapefile(data.flowline)

    assert len(shp) == 3694
    assert np.allclose(shp.centroid, [-107.57728379, 36.41307757])
    assert shp.ndim == 3
    assert shp.geometry_type == "3D MultiLineString"
    assert all([shape.type == "LineString" for shape in shp.shapes])


def test_geometry_write():
    data = ExampleData.NewMexico
    shp = tin.gis.load_shapefile(data.flowline)

    with TemporaryDirectory() as tmp_dir:
        shp.save("shapefile.shp")
        shp2 = tin.gis.load_shapefile("shapefile.shp")

        assert shp.crs == shp2.crs
        assert len(shp) == len(shp2)
        assert shp.geometry_type == shp2.geometry_type
        assert shp.ndim == shp2.ndim
        assert np.allclose(shp.extent, shp2.extent)
        assert shp.properties == shp2.properties


def test_raster_load():
    data = ExampleData.NewMexico
    dem = tin.gis.load_raster(data.dem)
    assert dem.shape == (2373, 2575)
    assert np.allclose(dem.extent, [-107.70482, 36.29657, -107.46639, 36.5163])

    # Test ASC (plain-text) files
    data = ExampleData.Borden
    _ = tin.gis.load_raster(data.dem_50cm)


def test_raster_write():
    with TemporaryDirectory() as tmp_dir:
        data = ExampleData.NewMexico
        dem = tin.gis.load_raster(data.dem)
        dem.save(os.path.join(tmp_dir, "test.tif"))

        dem2 = tin.gis.load_raster(os.path.join(tmp_dir, "test.tif"))

        assert dem.shape == dem2.shape
        assert np.allclose(dem.extent, dem2.extent)
        assert np.allclose(dem.masked_data(), dem2.masked_data())


def test_raster_boundary():
    data = ExampleData.NewMexico
    dem = tin.gis.load_raster(data.dem)
    boundary = dem.get_boundary()

    with TemporaryDirectory() as tmp_dir:
        boundary.save(os.path.join(tmp_dir, "test.shp"))

    _ = tin.gis.clip_raster(dem, boundary)

    assert True


def test_clip_raster():
    data = ExampleData.NewMexico
    dem = tin.gis.load_raster(data.dem)
    boundary = tin.gis.load_shapefile(data.watershed_boundary)
    new_dem = tin.gis.clip_raster(dem, boundary)
    new_dem.plot()
    assert new_dem.no_data_value == new_dem[0][0]
    assert round(np.nanmax(new_dem.data.data)) == 2279


def test_fill_raster_depressions():
    data = ExampleData.NewMexico
    dem = tin.gis.load_raster(data.dem)
    dem.fill_depressions()
    assert True


def test_reproject():
    data = ExampleData.NewMexico
    dem = tin.gis.load_raster(data.dem)
    boundary = tin.gis.load_shapefile(data.watershed_boundary)

    _ = tin.gis.reproject_raster(dem, "EPSG:32112")
    _ = tin.gis.reproject_geometry(boundary, "EPSG:32112")

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
            max_edge_length=0.1,
            scaling_type="relative",
            method=method,
            refinement_feature=flowline,
        )

        assert all(
            [x > 0.5 for x in tin.meshing.triangle_quality(surf)]
        ), f"Bad quality triangles ({method})"
        assert all(
            [x > 0.0 for x in tin.meshing.triangle_area(surf)]
        ), f"Zero or negative area triangles ({method})"


def test_save_mesh():
    data = ExampleData.Simple

    with TemporaryDirectory() as tmp_dir:
        surface_mesh = tin.meshing.load_mesh(data.surface_mesh)
        volume_mesh = tin.meshing.load_mesh(data.volume_mesh)

        for ext in ["vtk", "inp", "exo"]:
            surface_mesh.save(os.path.join(tmp_dir, f"surf.{ext}"))
            volume_mesh.save(os.path.join(tmp_dir, f"vol.{ext}"))
        # TODO: test that reading == saving


def test_point_sets():
    # tri_mesh =
    # vol_mesh =
    # surf_mesh = vol_mesh.surface_mesh()
    # top_points = surf_mesh.top_points
    # bottom_points = surf_mesh.bottom_points
    # side_points = surf_mesh.side_points
    # v1 = vol_mesh.points[top_points.primary_nodes]
    # assert np.in1d((v1.flatten(), tri_mesh.nodes.flatten()).all()
    # assert top_points == top_points.join(bottom_points).join(side_points).remove(bottom_points).remove(side_points)
    assert True


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
