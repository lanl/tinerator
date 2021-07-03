import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def get_cells_along_line(vtk_mesh, line_start, line_end):
    """Returns the cell IDs that intersect with the line formed by [line_start, line_end].

    Args:
        vtk_mesh ([type]): [description]
        line_start ([type]): [description]
        line_end ([type]): [description]

    Returns:
        [type]: [description]
    """
    cell_loc = vtk.vtkCellLocator()
    cell_loc.SetDataSet(vtk_mesh)
    cell_loc.BuildLocator()

    cell_ids = vtk.vtkIdList()
    cell_loc.FindCellsAlongLine(line_start, line_end, 0.001, cell_ids)

    return [cell_ids.GetId(i) for i in range(cell_ids.GetNumberOfIds())]


def check_orientation(mesh):
    """Returns True if nodes are ordered clockwise, and False otherwise.

    Args:
        mesh ([type]): [description]
    """

    raise NotImplementedError()

    # elements = mesh.elements - 1

    # edges = np.array([
    #    [elements[:,0], elements[:,1]],
    #    [elements[:,1], elements[:,2]],
    #    [elements[:,2], elements[:,3]],
    #    [elements[:,3], elements[:,0]],
    # ], dtype=int)

    # points = mesh.nodes[edges]

    # https://stackoverflow.com/a/1165943/5150303


def get_cell_normals(mesh, feature_angle: float = None):
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(mesh)

    if feature_angle is not None:
        normals.SetFeatureAngle(feature_angle)

    normals.ComputeCellNormalsOn()
    normals.ComputePointNormalsOff()
    normals.Update()

    output = normals.GetOutput().GetCellData().GetNormals()
    return vtk_to_numpy(output)


def prism_volume(mesh):
    """
    Computes the volume of each cell of a prism mesh.

    Args
    ----
        mesh (tinerator.meshing.Mesh): A TINerator mesh object.

    Returns
    -------
        An array of length ``n_elements`` containing the volume of
            each prism cell.
    """
    nodes = mesh.points
    prisms = mesh.elements - 1

    tris_bottom = prisms[:, :3]
    tris_top = prisms[:, 3:]

    def signed_area(nodes, tris):
        p1 = nodes[tris[:, 0]]
        p2 = nodes[tris[:, 1]]
        p3 = nodes[tris[:, 2]]

        return np.cross(p2 - p1, p3 - p2) / 2.0

    area_top = signed_area(nodes[:, :2], tris_top)
    area_bottom = signed_area(nodes[:, :2], tris_bottom)
    area_mean = np.mean([area_top, area_bottom], axis=0)

    z = nodes[:, 2]
    z_top = z[tris_top]
    z_bottom = z[tris_bottom]

    cell_depth = np.mean(z_top - z_bottom, axis=1)

    return area_mean * cell_depth


def triangle_quality(mesh):
    """
    Returns the quality of each triangle in the surface mesh,
    where quality is defined as twice the ratio of circumcircle
    and incircle radius.

    Args
    ----
        mesh (tinerator.meshing.Mesh): A triangular TINerator mesh.

    Returns
    -------
        An array of length ``n_elements``, with quality value for
        each tri.
    """

    def circumcircle(a, b, c):
        # a, b, c are lengths of the three sides of a triangle
        # https://www.mathopenref.com/trianglecircumcircle.html
        return (a * b * c) / np.sqrt(
            (a + b + c) * (b + c - a) * (c + a - b) * (a + b - c)
        )

    def incircle(a, b, c):
        # a, b, c are lengths of the three sides of a triangle
        s = (a + b + c) / 2.0
        return np.sqrt(((s - a) * (s - b) * (s - c)) / s)

    a, b, c = triangle_edge_lengths(mesh)
    return 2.0 * incircle(a, b, c) / circumcircle(a, b, c)


def triangle_edge_lengths(mesh):
    """
    Computes the edge length for each edge in each triangle.

    Args
    ----
        mesh (tinerator.meshing.Mesh): A trianglar TINerator mesh.

    Returns
    -------
        A tuple of arrays, containing the edge lengths for all
        three edges in each triangle.
    """
    pts = mesh.nodes[:, :2]
    elems = mesh.elements - 1

    v1 = pts[elems[:, 0]]
    v2 = pts[elems[:, 1]]
    v3 = pts[elems[:, 2]]

    a = np.linalg.norm(v1 - v2, axis=1)
    b = np.linalg.norm(v2 - v3, axis=1)
    c = np.linalg.norm(v3 - v1, axis=1)

    return (a, b, c)


def triangle_area(mesh):
    """
    Returns the signed area for all triangles in
    the mesh.

    Args
    ----
        mesh (tinerator.meshing.Mesh): A TINerator triangular mesh.

    Returns
    -------
        An array of length ``n_elements`` containing the signed area
            of every triangle.
    """
    nodes = mesh.nodes[:, :2]
    triangles = mesh.elements - 1

    p1 = nodes[triangles[:, 0]]
    p2 = nodes[triangles[:, 1]]
    p3 = nodes[triangles[:, 2]]

    return np.cross(p2 - p1, p3 - p2) / 2.0


def edge_lengths(mesh) -> np.ndarray:
    """
    Returns an array with the Euclidean length of each edge
    in `mesh.edges`.
    """

    pts = mesh.nodes
    edges = mesh.edges - 1
    return np.linalg.norm(pts[edges[:, 0]] - pts[edges[:, 1]], axis=1)
