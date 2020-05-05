import vtk
import numpy as np
import pyvista as pv

def plot_3d(mesh,element_type:str,**kwargs):
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

    # plot the grid
    grid.plot(show_edges=True,show_bounds=True,**kwargs)
