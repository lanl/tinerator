import vtk
import numpy as np
import pyvista as pv

def plot_3d(mesh,element_type:str,cell_arrays:dict=None,node_arrays:dict=None):
    '''
    TODO: this shouldn't directly depend on the mesh class.
    TODO: add prism support.
    TODO: add attribute support.
    '''


    if element_type.lower() == 'tri':
        nodes_per_elem = 3
        vtk_cell_type = vtk.VTK_TRIANGLE
    else:
        raise ValueError("Unsupported element type")
    
    ncells = mesh.n_elements
    nnodes = mesh.n_nodes

    offset = np.array([(nodes_per_elem+1)*i for i in range(ncells)])
    cells = np.hstack((np.full((ncells,1),nodes_per_elem), mesh.elements - 1)).flatten()
    cell_type = np.repeat([vtk_cell_type],ncells)
    nodes = mesh.nodes

    # create the unstructured grid directly from the numpy arrays
    grid = pv.UnstructuredGrid(offset, cells, cell_type, nodes, deep=True)

    # https://github.com/pyvista/pyvista/blob/52c78d610c30f5f7e02dacecbb081211faae8073/docs/getting-started/what-is-a-mesh.rst
    
    scalar = None

    if cell_arrays:
        for key in cell_arrays:
            grid.cell_arrays[key] = cell_arrays[key]
            scalar = key

    if node_arrays:
        for key in node_arrays:
            grid.node_arrays[key] = node_arrays[key]
            scalar = key

    # plot the grid
    grid.plot(scalars=scalar,show_edges=True,show_bounds=True)
