import sys
sys.path.insert(0,'/Users/livingston/playground/tinerator/tinerator-core')
import os
import tinerator as tin
from helper import DATA_DIR

MESH_DIR = os.path.join(DATA_DIR, 'meshes')

def test_mpas():

    mesh_vtk = tin.meshing.load(os.path.join(MESH_DIR, 'mpas', 'delaware_local_2d.vtk'))
    mesh_mpas = tin.meshing.load(os.path.join(MESH_DIR, 'mpas', 'delaware_local_2d_mpas.nc'))

    assert mesh_vtk.n_nodes == mesh_mpas.n_nodes, 'Node count differed'
    assert mesh_vtk.n_elements == mesh_mpas.n_elements, 'Element count different'
    
    assert np.array_equal(mesh_vtk.nodes, mesh_mpas.nodes), 'Nodes array differs'
    assert np.array_equal(mesh_vtk.elements, mesh_mpas.elements), 'Element array differs'