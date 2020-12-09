import vtk
import numpy as np
import pyvista as pv
from copy import deepcopy


def plot_3d(
    mesh,
    element_type: str,
    cell_arrays: dict = None,
    node_arrays: dict = None,
    scale: tuple = (1, 1, 1),
    window_size: tuple = (1024, 1080),
    show_bounds: bool = True,
    show_edges: bool = True,
    title: str = "",
    show_mesh_info: bool = True,
    show_axes: bool = True,
    cmap: str = "gist_earth_r",
    savefig: str = None,
    edge_color: tuple = (0.6, 0.6, 0.6),
    notebook: bool = False,
):

    """
    TODO: add texturing
        texture : vtk.vtkTexture or np.ndarray or boolean, optional
            A texture to apply if the input mesh has texture
            coordinates.  This will not work with MultiBlock
            datasets. If set to ``True``, the first available texture
            on the object will be used. If a string name is given, it
            will pull a texture with that name associated to the input
            mesh.
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
        cells = np.hstack(
            (delta.reshape((delta.shape[0], 1)) + 1, mesh.elements)
        )
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
    grid = pv.UnstructuredGrid(offset, cells, cell_type, nodes, deep=True)

    # TODO: approach this in a different way
    scalar = None
    if cell_arrays:
        for key in cell_arrays:
            grid.cell_arrays[key] = cell_arrays[key]
            scalar = key

    if node_arrays:
        for key in node_arrays:
            grid.point_arrays[key] = node_arrays[key]
            scalar = key

    # Plot the grid
    if notebook:
        # Requires vtk==8.1.2 ?
        import pyvistaqt as pvqt

        plotter = pvqt.BackgroundPlotter()
    else:
        plotter = pv.Plotter()

    plotter.add_mesh(
        grid,
        scalars=scalar,
        show_edges=show_edges,
        cmap=cmap,
        edge_color=edge_color,
    )

    if title is not None:
        plotter.add_text(title, position="upper_right", shadow=True)

    if show_mesh_info:
        plotter.add_text(
            f"Nodes: {nnodes}\nCells: {ncells}\nElement type: {element_type}",
            position="upper_left",
            shadow=True,
            font_size=12,
        )

    if show_axes:
        plotter.show_bounds(
            grid="back", location="outer", ticks="both"
        )  # , font_size=20)
        plotter.add_axes(interactive=True, line_width=4)

    if savefig is not None:
        screenshot = savefig
        interactive = False
    else:
        screenshot = False
        interactive = True

    if not notebook:
        plotter.show(
            title="TINerator",
            window_size=window_size,
            screenshot=screenshot,
            interactive=interactive,
        )
