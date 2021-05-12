import vtk
import numpy as np
import pyvista as pv
from copy import deepcopy
from ..constants import _in_notebook, JUPYTER_BACKEND_DEFAULT

def _mesh_to_vtk_unstructured(mesh, element_type: str, scale: tuple = (1, 1, 1), cell_arrays = None, node_arrays = None):
    """
    Internal function.
    Converts a TINerator Mesh object into a VTK unstructured grid.
    """
    ncells = mesh.n_elements
    nnodes = mesh.n_nodes

    if not ncells:
        raise ValueError("Cells must be defined to visualize")

    if element_type.lower() == "tri":
        nodes_per_elem = 3
        vtk_cell_type = vtk.VTK_TRIANGLE
    elif element_type.lower() == "prism":
        nodes_per_elem = 6
        vtk_cell_type = vtk.VTK_WEDGE
    elif element_type.lower() == "polygon":
        vtk_cell_type = vtk.VTK_POLYGON
    else:
        raise ValueError("Unsupported element type")

    if vtk_cell_type == vtk.VTK_POLYGON:
        delta = np.count_nonzero(mesh.elements, axis=1)
        offset = np.array([np.sum(delta[:i] + 1) for i in range(len(delta))])
        cells = np.hstack((delta.reshape((delta.shape[0], 1)) + 1, mesh.elements))
        cells = cells[cells > 0] - 1
    else:
        offset = np.array([(nodes_per_elem + 1) * i for i in range(ncells)])
        cells = np.hstack(
            (np.full((ncells, 1), nodes_per_elem), mesh.elements - 1)
        ).flatten()

    cell_type = np.repeat([vtk_cell_type], ncells)
    nodes = deepcopy(mesh.nodes)

    # Scale mesh coordinates
    for i in range(3):
        nodes[:, i] = scale[i] * nodes[:, i]

    # Create the unstructured grid directly from the numpy arrays
    if vtk.VTK_MAJOR_VERSION >= 9:
        grid = pv.UnstructuredGrid(cells, cell_type, nodes, deep=True)
    else:
        grid = pv.UnstructuredGrid(offset, cells, cell_type, nodes, deep=True)

    if cell_arrays is not None:
        for key in cell_arrays:
            grid.cell_arrays[key] = cell_arrays[key]

    if node_arrays is not None:
        for key in node_arrays:
            grid.point_arrays[key] = node_arrays[key]

    return grid


def plot_3d(
    mesh,
    element_type: str,
    active_scalar: str = None,
    cell_arrays: dict = None,
    node_arrays: dict = None,
    scale: tuple = (1, 1, 1),
    window_size: tuple = (1024, 1080),
    show_bounds: bool = True,
    show_edges: bool = True,
    text: str = None,
    show_axes: bool = True,
    cmap: str = "gist_earth_r",
    screenshot: str = None,
    edge_color: tuple = (0.6, 0.6, 0.6),
    jupyter_notebook: bool = _in_notebook(),
    **kwargs
):
    """
    Uses PyVista and VTK to render an unstructured mesh
    in a live window or Jupyter notebook cell.
    """

    # Additional documentation:
    # https://docs.pyvista.org/plotting/plotting.html#pyvista.plot
    # https://docs.pyvista.org/user-guide/jupyter/ipygany.html#returning-scenes

    grid = _mesh_to_vtk_unstructured(
        mesh, 
        element_type, 
        scale=scale, 
        cell_arrays=cell_arrays, 
        node_arrays=node_arrays
    )

    if jupyter_notebook:
        jupyter_backend = JUPYTER_BACKEND_DEFAULT
    else:
        jupyter_backend = None

    grid.plot(
        screenshot=screenshot,
        window_size=window_size,
        jupyter_backend=jupyter_backend,
        text=text,
        show_axes=show_axes,
        show_bounds=show_bounds,
        show_edges=show_edges,
        edge_color=edge_color,
        cmap=cmap,
        scalars=active_scalar,
        **kwargs
    )
