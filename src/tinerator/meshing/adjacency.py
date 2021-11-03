"""
Various classes and functions for constructing adjacency matrices
and graphs from meshes.
"""

import vtk
import numpy as np
from typing import Union, List
from scipy.spatial import distance
from scipy.sparse import lil_matrix, coo_matrix, triu
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from pyvista.utilities.helpers import vtk_id_list_to_array as vtkID_to_arr


def adjacency_matrix(
    mesh: vtk.vtkPolyData,
    upper_triang: bool = False,
    as_numpy: bool = False,
    build_links: bool = True,
) -> Union[lil_matrix, coo_matrix, np.ndarray]:
    """
    Generates an adjacency matrix showing which mesh cells share faces/edges.
    The matrix is (num_cells, num_cells) in size, and the value at [cell_i, cell_j]
    is ``True`` if they are adjacent and ``False`` if not.

    Args:
        mesh (vtk.vtkPolyData): The VTK mesh to compute on.
        upper_triang (bool, optional): If True, returns only the upper triangular of the sparse matrix. Defaults to False.
        as_numpy (bool, optional): If True, returns the matrix in NumPy format. Defaults to False.
        build_links (bool, optional): Links are required to get the edge graph. Defaults to True.

    Returns:
        Union[lil_matrix, coo_matrix, np.ndarray]: The sparse matrix.
    """

    num_cells = mesh.GetNumberOfCells()

    # VTK will crash on `GetCellEdgeNeighbors` if links are not built
    if build_links:
        mesh.BuildLinks(0)

    sparse_matrix = lil_matrix((num_cells, num_cells), dtype=bool)
    cell_neighbors = vtk.vtkIdList()

    # Iterate over each cell and each edge within that cell
    # Use the `GetCellEdgeNeighbors` method to get the adjacency of
    # a single cell, and use that adjacency to construct the sparse matrix
    for cell_id in range(num_cells):
        cell = mesh.GetCell(cell_id)

        for edge_id in range(cell.GetNumberOfEdges()):
            cell_neighbors.Initialize()
            edge = vtkID_to_arr(cell.GetEdge(edge_id).GetPointIds())
            mesh.GetCellEdgeNeighbors(cell_id, edge[0], edge[1], cell_neighbors)

            for neighbor in vtkID_to_arr(cell_neighbors):
                sparse_matrix[cell_id, neighbor] = True

    if upper_triang:
        sparse_matrix = triu(sparse_matrix)

    if as_numpy:
        sparse_matrix = sparse_matrix.todense()

    return sparse_matrix


class UndirectedGraph(vtk.vtkMutableUndirectedGraph):
    """
    This represents a (VTK-format) mesh in undirected graph form.
    This is useful for determining the adjacency of mesh elements,
    and to quickly iterate across cells with shared faces.
    """

    def __init__(self, mesh: vtk.vtkPolyData):
        super().__init__()
        sparse_matrix = adjacency_matrix(
            mesh, upper_triang=True, as_numpy=True, build_links=True
        )

        for _ in range(mesh.GetNumberOfCells()):
            self.AddVertex()

        for conn in np.argwhere(sparse_matrix):
            self.AddEdge(*conn)

        self.GetPoints().SetData(numpy_to_vtk(mesh.cell_centers().points))

    @property
    def num_nodes(self) -> int:
        """
        Returns the number of graph nodes. This should be exactly
        equal to the number of cells in the parent mesh.
        """
        return self.GetNumberOfVertices()

    @property
    def num_edges(self) -> int:
        """
        Returns the number of edges in the graph. This corresponds to the number
        of cells with shared edges/faces in the mesh.
        """
        return self.GetNumberOfEdges()

    @property
    def nodes(self) -> np.ndarray:
        """
        Returns the (x, y, z) points of the graph nodes.
        These represent the cell centers in the parent mesh.
        """
        return vtk_to_numpy(self.GetPoints().GetData())

    @property
    def edges(self) -> List[List[int]]:
        """
        Returns all edges in the graph, in the form: [source_cell, target_cell]
        """
        all_edges = []

        it = vtk.vtkEdgeListIterator()
        self.GetEdges(it)

        while it.HasNext():
            edge = it.NextGraphEdge()
            all_edges.append([edge.GetSource(), edge.GetTarget()])

        return all_edges

    def get_adjacent_nodes(self, node_id: int) -> List[int]:
        """Returns the graph nodes adjacent to ``node_id``.

        Args:
            node_id (int): The graph node to query.

        Returns:
            List[int]: The adjacent graph nodes to ``node_id``.
        """
        nodes = []
        it = vtk.vtkAdjacentVertexIterator()
        self.GetAdjacentVertices(node_id, it)

        while it.HasNext():
            edge = it.Next()
            nodes.append(int(edge))

        return nodes

    def plot(self) -> None:
        """
        Plots the undirected graph in a VTK window.
        Headless framebuffer support, including in Jupyter Notebooks,
        is not currently supported.
        """
        graphLayoutView = vtk.vtkGraphLayoutView()
        graphLayoutView.AddRepresentationFromInput(self)
        graphLayoutView.ResetCamera()
        graphLayoutView.Render()
        graphLayoutView.GetInteractor().Start()
