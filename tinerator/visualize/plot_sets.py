import numpy as np
import pyvista as pv
from ..constants import (
    is_tinerator_object,
    _in_docker_container,
    _init_pyvista_framebuffer,
)

# from ..meshing import PointSet, SideSet, ElementSet


def plot_sets(mesh, sets, num_cols: int = 3, link_views: bool = True):
    """
    Plots the mesh along with element sets, side (face) sets, and
    point sets.
    """
    if _in_docker_container():
        _init_pyvista_framebuffer()

    if not isinstance(sets, (list, tuple, np.ndarray)):
        sets = [sets]

    num_subplots = len(sets) + 1
    if num_subplots <= 3:
        num_rows = 1
        num_cols = num_subplots
    else:
        num_rows = int(np.ceil(num_subplots / num_cols))

    p = pv.Plotter(shape=(num_rows, num_cols))

    for (i, mesh_obj) in enumerate([mesh, *sets]):
        kwargs = {}
        p.subplot(i // num_cols, i % 3)

        if is_tinerator_object(mesh_obj, "PointSet"):  # isinstance(mesh_obj, PointSet):
            mesh_name = f'("{mesh_obj.name}")' if mesh_obj.name is not None else ""
            mesh_name = f"Point Set {mesh_name}".strip()
            mesh_obj = mesh_obj.to_vtk_mesh()
            kwargs["color"] = "red"
            kwargs["render_points_as_spheres"] = True
        elif is_tinerator_object(mesh_obj, "SideSet"):
            mesh_name = f'("{mesh_obj.name}")' if mesh_obj.name is not None else ""
            mesh_name = f"Side Set {mesh_name}".strip()
            mesh_obj = mesh_obj.to_vtk_mesh()
        elif is_tinerator_object(mesh_obj, "ElementSet"):
            raise NotImplementedError()
        else:
            mesh_name = "Primary Mesh"
            kwargs["show_edges"] = True

        p.add_text(mesh_name, font_size=12)
        p.add_mesh(mesh_obj, **kwargs)

    if link_views:
        p.link_views()

    p.show()
