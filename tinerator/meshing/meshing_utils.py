import math
from math import sqrt
import numpy as np


def ravel_faces_to_vtk(faces):
    """
    Ravels an NxM matrix, or list of lists, containing
    face information into a VTK-compliant 1D array.
    """
    vtk_faces = []
    for face in faces:
        nan_mask = np.isnan(face)

        if nan_mask.any():
            face = np.array(face)[~nan_mask]

        vtk_face = np.hstack([len(face), face])
        vtk_faces.extend(vtk_face)

    return np.hstack(vtk_faces).astype(int)


def unravel_vtk_faces(faces_vtk, fill_matrix: bool = False):
    """
    Unravels VTK faces. If fill_matrix = True,
    then instead of returning a list of unequal length
    arrays, it returns an NxM **floating-point** array with unequal
    rows filled with ``numpy.nan``.
    """
    faces = []

    i = 0
    sz_faces = len(faces_vtk)
    max_stride = -1
    while True:
        if i >= sz_faces:
            break
        stride = faces_vtk[i]
        max_stride = max(max_stride, stride)
        j = i + stride + 1
        faces.append(faces_vtk[i + 1 : j])
        i = j

    if fill_matrix:
        faces = [
            np.hstack([face, [np.nan] * (max_stride - len(face))]) for face in faces
        ]
        return np.array(faces)
    else:
        return np.array(faces, dtype=object)


def refit_arrays(arr1, arr2, type="float64"):
    """
    Expands the shape of the array to ``target_shape`` and
    fills extra rows/cols with ``np.nan``.
    Useful for getting two arrays to fit to the same shape.
    """

    def resize(arr, target_shape, type="float64"):
        shape_delta = tuple(np.array(target_shape) - np.array(arr.shape))
        padding = ((0, shape_delta[0]), (0, shape_delta[1]))
        return np.pad(arr.astype(type), padding, constant_values=np.nan)

    target_shape = np.max([arr1.shape, arr2.shape], axis=0)
    return resize(arr1, target_shape, type=type), resize(arr2, target_shape, type=type)


def in2d(a: np.ndarray, b: np.ndarray, assume_unique: bool = False) -> np.ndarray:
    """
    Helper function to replicate numpy.in1d, but with
    NxM matrices.
    """
    # https://stackoverflow.com/a/16216866

    def as_void(arr):
        arr = np.ascontiguousarray(arr)
        if np.issubdtype(arr.dtype, np.floating):
            arr += 0.0
        return arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[-1])))

    init_shape = a.shape
    a, b = refit_arrays(a, b, type="float64")
    return np.in1d(as_void(a), as_void(b), assume_unique)[: init_shape[0]]


def is_geometry(obj):
    try:
        return (
            obj.__module__ == "tinerator.gis.geometry"
            and type(obj).__name__ == "Geometry"
        )
    except:
        return False


def clockwiseangle_and_distance(point, origin=[0, 0], refvec=[0, 1]):
    # https://stackoverflow.com/a/41856340/5150303
    # Vector between point and the origin: v = p - o
    vector = [point[0] - origin[0], point[1] - origin[1]]
    # Length of vector: ||v||
    lenvector = math.hypot(vector[0], vector[1])
    # If length is zero there is no angle
    if lenvector == 0:
        return -math.pi, 0
    # Normalize vector: v/||v||
    normalized = [vector[0] / lenvector, vector[1] / lenvector]
    dotprod = normalized[0] * refvec[0] + normalized[1] * refvec[1]  # x1*x2 + y1*y2
    diffprod = refvec[1] * normalized[0] - refvec[0] * normalized[1]  # x1*y2 - y1*x2
    angle = math.atan2(diffprod, dotprod)
    # Negative angles represent counter-clockwise angles so we need to subtract them
    # from 2*pi (360 degrees)
    if angle < 0:
        return 2 * math.pi + angle, lenvector
    # I return first the angle because that's the primary sorting criterium
    # but if two vectors have the same angle then the shorter distance should come first.
    return angle, lenvector


def get_linestring_connectivity(
    nodes: np.array, closed: bool = True, clockwise: bool = True
) -> np.ndarray:
    """
    Internal function. Returns a connectivity array
    for an ordered array of nodes.
    Assumes that all points are ordered clockwise.
    """
    closed_conn = [(len(nodes), 1)] if closed else []
    connectivity = np.array(
        [(i, i + 1) for i in range(1, len(nodes))] + closed_conn, dtype=int
    )

    return connectivity


def estimate_edge_lengths(raster, n_tris: int) -> float:
    """
    Estimates the edge lengths of `n_tris` amount of equilateral triangles
    that would fit into `raster`.
    """
    return sqrt(4.0 / sqrt(3.0) * (raster.area / float(n_tris)))
