import vtk
import numpy as np
import pyvista as pv
from copy import deepcopy

def plot_3d(mesh,element_type:str,cell_arrays:dict=None,node_arrays:dict=None,scale:tuple=(1,1,1), **kwargs):
    '''
    See `help(pyvista.UnstructuredGrid.plot)` and `help(pyvista.Plotter.add_mesh)`
    for more information on possible keyword arguments.
    '''

    '''
    TODO: this shouldn't directly depend on the mesh class.
    TODO: add texturing
        texture : vtk.vtkTexture or np.ndarray or boolean, optional
            A texture to apply if the input mesh has texture
            coordinates.  This will not work with MultiBlock
            datasets. If set to ``True``, the first available texture
            on the object will be used. If a string name is given, it
            will pull a texture with that name associated to the input
            mesh.
    '''


    if element_type.lower() == 'tri':
        nodes_per_elem = 3
        vtk_cell_type = vtk.VTK_TRIANGLE
    elif element_type.lower() == 'prism':
        nodes_per_elem = 6
        vtk_cell_type = vtk.VTK_WEDGE
    else:
        raise ValueError("Unsupported element type")
    
    ncells = mesh.n_elements
    nnodes = mesh.n_nodes

    offset = np.array([(nodes_per_elem+1)*i for i in range(ncells)])
    cells = np.hstack((np.full((ncells,1),nodes_per_elem), mesh.elements - 1)).flatten()
    cell_type = np.repeat([vtk_cell_type],ncells)
    nodes = deepcopy(mesh.nodes)

    # Scale mesh coordinates
    for i in range(3):
        nodes[:,i] = scale[i]*nodes[:,i]

    # create the unstructured grid directly from the numpy arrays
    grid = pv.UnstructuredGrid(offset, cells, cell_type, nodes, deep=True)

    # https://github.com/pyvista/pyvista/blob/52c78d610c30f5f7e02dacecbb081211faae8073/docs/getting-started/what-is-a-mesh.rst
    
    scalar = None

    # TODO: approach this in a different way
    if cell_arrays:
        for key in cell_arrays:
            grid.cell_arrays[key] = cell_arrays[key]
            scalar = key

    if node_arrays:
        for key in node_arrays:
            grid.node_arrays[key] = node_arrays[key]
            scalar = key

    # plot the grid
    grid.plot(scalars=scalar,show_edges=True,show_bounds=True, **kwargs)
