import os
import tinerator as tin
import numpy as np
from helper import MESH_DIR


def test_avs():
    mesh_avs = tin.meshing.load(os.path.join(MESH_DIR, "avs", "simple_surface.inp"))

    assert mesh_avs.n_nodes == 9, "Node count is wrong"
    assert mesh_avs.n_elements == 8, "Element count is wrong"
    assert (
        mesh_avs.element_type == tin.meshing.ElementType.TRIANGLE
    ), "Element type is wrong"


def test_mpas():

    mesh_vtk = tin.meshing.load(os.path.join(MESH_DIR, "mpas", "delaware_local_2d.vtk"))
    mesh_mpas = tin.meshing.load(
        os.path.join(MESH_DIR, "mpas", "delaware_local_2d_mpas.nc")
    )

    assert mesh_vtk.n_nodes == mesh_mpas.n_nodes, "Node count differed"
    assert mesh_vtk.n_elements == mesh_mpas.n_elements, "Element count different"

    assert np.array_equal(mesh_vtk.nodes, mesh_mpas.nodes), "Nodes array differs"
    assert np.array_equal(
        mesh_vtk.elements, mesh_mpas.elements
    ), "Element array differs"


def test_mesh_from_arrays():
    """Simple triangle surface mesh"""

    nodes = np.array(
        [
            [1.000000000000e01, 1.000000000000e01, 1.000000000000e01],
            [2.000000000000e01, 1.000000000000e01, 1.000000000000e01],
            [2.000000000000e01, 2.000000000000e01, 1.000000000000e01],
            [1.000000000000e01, 2.000000000000e01, 1.000000000000e01],
            [1.500000000000e01, 1.500000000000e01, 1.000000000000e01],
        ],
        dtype=float,
    )

    connectivity = np.array(
        [
            [1, 5, 4],
            [5, 3, 4],
            [1, 2, 5],
            [5, 2, 3],
        ],
        dtype=int,
    )

    mesh = tin.meshing.Mesh(
        nodes=nodes, elements=connectivity, etype=tin.meshing.ElementType.TRIANGLE
    )

    assert mesh.n_nodes == 5, "Node count differs"
    assert mesh.n_elements == 4, "Element count differs"
    assert (
        mesh.element_type == tin.meshing.ElementType.TRIANGLE
    ), "Element type is wrong"
