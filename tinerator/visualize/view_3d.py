import pyvista as pv
import vtk

def plot_3d(mesh,**kwargs):
    '''TODO: be able to handle multiple meshes. Useful for debugging.'''
    # offset array.  Identifies the start of each cell in the cells array
    offset = np.array([0, 9])

    # Contains information on the points composing each cell.
    # Each cell begins with the number of points in the cell and then the points
    # composing the cell
    cells = np.array([8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 10, 11, 12, 13, 14, 15])

    # cell type array. Contains the cell type of each cell
    cell_type = np.array([vtk.VTK_HEXAHEDRON, vtk.VTK_HEXAHEDRON])

    cell1 = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ]
    )

    cell2 = np.array(
        [
            [0, 0, 2],
            [1, 0, 2],
            [1, 1, 2],
            [0, 1, 2],
            [0, 0, 3],
            [1, 0, 3],
            [1, 1, 3],
            [0, 1, 3],
        ]
    )

    # points of the cell array
    points = np.vstack((cell1, cell2))

    # create the unstructured grid directly from the numpy arrays
    grid = pv.UnstructuredGrid(offset, cells, cell_type, points)

    # plot the grid
    grid.plot(show_edges=True,show_bounds=True,**kwargs)