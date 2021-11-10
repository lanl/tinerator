import numpy as np
import triangle as tr
from .mesh import Mesh
from .meshing_types import ElementType
from ..gis import unproject_vector


def get_edges(tris: np.ndarray) -> np.ndarray:
    """
    Gets the edges from a triangular connectivity array.
    """
    all_edges = np.vstack(
        (
            tris[:, :2],
            tris[:, 1:3],
            np.transpose(np.array([tris[:, 0], tris[:, 2]])),
        )
    )
    edges = []

    for edge in all_edges:
        edges.append([min(edge), max(edge)])

    return np.unique(edges, axis=0)


def get_refined_triplane(
    dem,
    distance_map,
    min_edge: float,
    max_edge: float,
    min_dist: float,
    max_dist: float,
    n_iterations: int = 5,
) -> Mesh:
    """
    Creates a triplane where

             |
    maxEdge  |           /-------
             |         /
             |       /
             |     /
    minEdge  |----
             |
             |___________________
                  ^      ^
                  |      |
            minDist      maxDist


    """

    max_area = max_edge  # 0.5 * max_edge**2.
    assert (
        dem.data.shape == distance_map.shape
    ), "DEM and distance map must have the same shape"

    # CONVERT THE DISTANCE MAP TO AN EDGE LENGTHS MAP
    # Each index of this matrix will contain what the
    # triangle edge length should be at that point

    # Later, we will split edges at their midpoints
    # based on how their existing edges compare to the
    # matrix `A`
    A = np.zeros(distance_map.shape)

    for row in range(distance_map.shape[0]):
        for col in range(distance_map.shape[1]):

            x = distance_map[row][col]

            if x < min_dist:
                y = min_edge
            elif x > max_dist:
                y = max_edge
            else:
                m = (max_edge - min_edge) / (max_dist - min_dist)
                y = m * (x - min_dist) + min_edge

            A[row][col] = y

    boundary_len = max_edge * 0.2  # 5.
    vertices, connectivity = dem.get_boundary(boundary_len, connect_ends=True)
    # vertices = unproject_vector(vertices, dem)

    plt.imshow(A)
    plt.cbar()
    plt.show()

    # Create an initial 'uniform' triangular mesh
    t = tr.triangulate(
        {
            "vertices": list(vertices[:, :2]),
            "segments": list(connectivity - 1),
        },
        # p enforces boundary connectivity,
        # q gives a quality mesh,
        # and aX is max edge length
        "pqa%f" % (round(max_area, 2)),
    )

    for _ in range(n_iterations):

        new_points = []

        edges = get_edges(t["triangles"])
        vertices = t["vertices"]

        for edge in edges:

            verts = vertices[edge]
            p0, p1 = verts

            midpoint = np.mean(verts, axis=0)
            edge_length = np.sum((p1 - p0) ** 2) ** 0.5

            if edge_length > A[int(midpoint[1])][int(midpoint[0])]:
                new_points.append(midpoint)

        vertices = list(t["vertices"])
        vertices.extend(list(new_points))

        print(1)

        t = tr.triangulate(
            {"vertices": t["vertices"], "segments": list(connectivity - 1)},
            "pqa%f" % (round(max_area, 2)),
        )

        print(2)

    m = Mesh()
    m.nodes = np.hstack((t["vertices"], np.zeros((t["vertices"].shape[0], 1))))
    m.elements = t["triangles"] + 1
    m.element_type = ElementType.TRIANGLE

    # z_values = map_elevation(dem, m.nodes)
    # m.nodes[:,2] = z_values

    return m
