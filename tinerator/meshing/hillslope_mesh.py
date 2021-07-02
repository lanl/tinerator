import numpy as np
from pyproj import CRS
from ..logging import warn, debug
from .meshing_types import ElementType
from .mesh import Mesh


def create_hillslope_mesh(
    z_data: np.ndarray,
    x_coords: np.ndarray = None,
    y_coords: np.ndarray = None,
    crs: CRS = None,
):
    """
    Generates a hillslope mesh (structured grid) from an NxM NumPy array representing
    each cell of the grid.

    Args:
        z_data (np.ndarray): [description]
        x_coords (np.ndarray, optional): [description]. Defaults to None.
        y_coords (np.ndarray, optional): [description]. Defaults to None.
    """
    if not isinstance(z_data, np.ndarray):
        z_data = np.array(z_data)

    try:
        _ = z_data.shape[1]
    except IndexError:
        z_data = z_data[None]

    if z_data.shape[0] == 1:
        z_data = np.vstack((z_data, z_data))
    if z_data.shape[1] == 1:
        z_data = np.hstack((z_data, z_data))

    z_rows, z_cols = z_data.shape

    if x_coords is None:
        x_coords = np.arange(0, z_cols, 1)

    if y_coords is None:
        y_coords = np.arange(0, z_rows, 1)

    xx, yy = np.meshgrid(x_coords, y_coords)
    coords = np.vstack([xx.ravel(), yy.ravel(), z_data.ravel()]).T

    quads = []
    for j in range(z_rows - 1):
        for i in range(z_cols - 1):
            quad = [
                i + j * z_cols,
                i + j * z_cols + 1,
                i + 1 + (j + 1) * z_cols,
                i + (j + 1) * z_cols,
            ]
            quads.append(quad)

    quads = np.array(quads, dtype=int) + 1
    mesh = Mesh(
        nodes=coords.astype(float),
        elements=quads.astype(int),
        etype=ElementType.QUAD,
        crs=crs,
    )

    mesh.material_id = 1

    warn("Hillslope and quad meshes are an experimental feature.")
    warn("Expect things to break.")

    return mesh
