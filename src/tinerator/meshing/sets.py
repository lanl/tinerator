import pyvista as pv
import numpy as np
from .meshing_utils import (
    convert_vtk_faces_to_exodusii,
    ravel_faces_to_vtk,
    unravel_vtk_faces,
    in2d,
    refit_arrays,
    convert_vtk_faces_to_exodusii,
)


class SideSet(object):
    """
    SideSet object.
    """

    def __init__(
        self,
        primary_mesh,
        primary_cells,
        primary_faces,
        name: str = None,
        set_id: int = None,
    ):
        self.name = name
        self.primary_mesh = primary_mesh
        self.primary_cells = primary_cells
        self.primary_faces = primary_faces
        self.set_id = set_id

    def __repr__(self):
        return f"SideSet<name={self.name}, num_faces={len(self.primary_cells)}>"

    def save(self, outfile: str):
        self.to_vtk_mesh().save(outfile)

    def to_vtk_mesh(self):
        pts = self.primary_mesh.points
        faces = self.primary_faces
        return pv.PolyData(pts, faces)

    @property
    def id(self):
        return self.set_id

    @id.setter
    def id(self, value):
        assert isinstance(value, int)
        self.set_id = value

    @property
    def exodusii_format(self):
        """
        Returns set as a dictionary in an ExodusII-compliant schema.
        """

        cells = unravel_vtk_faces(self.primary_mesh.cells)[self.primary_cells]
        cell_types = self.primary_mesh.celltypes[self.primary_cells]
        faces = convert_vtk_faces_to_exodusii(
            self.primary_faces,
            cells,
            cell_types,
            unravel_faces=True,
            unravel_cells=False,
        )

        num_faces = len(faces)

        assert num_faces == len(self.primary_cells), "Num. face and num. cell mismatch"

        return {
            "set_name": self.name,
            "set_id": self.set_id,
            "set_elems": self.primary_cells + 1,
            "set_sides": faces,
            "num_set_sides": num_faces,
            "num_set_dist_facts": 0,
        }

    def view(self, **kwargs):
        """
        Renders a 3D visualization of the set.

        Args
        ----
            **kwargs: Keyword arguments to pass to PyVista.
        """
        self.to_vtk_mesh().plot(**kwargs)

    def _ufuncs(self, side_set, type: str, set_name: str = None):
        """
        Master function for performing set operations.
        """
        assert (
            self.primary_mesh == side_set.primary_mesh
        ), "Sets must have the same parent mesh"

        if set_name is None:
            set_name = self.name

        this_cells = self.primary_cells
        that_cells = side_set.primary_cells

        this_faces = unravel_vtk_faces(self.primary_faces, fill_matrix=True)
        that_faces = unravel_vtk_faces(side_set.primary_faces, fill_matrix=True)

        mask = in2d(this_faces, that_faces) & np.in1d(this_cells, that_cells)

        if type in ["join", "union"]:
            this_faces, that_faces = refit_arrays(this_faces[~mask], that_faces)
            new_cells = np.hstack([this_cells[~mask], that_cells])
            new_faces = np.vstack([this_faces, that_faces])
            new_faces = new_faces[~np.all(np.isnan(new_faces), axis=1)]
        elif type in ["intersect", "intersection"]:
            new_cells = this_cells[mask]
            new_faces = this_faces[mask]
        elif type in ["remove", "filter", "diff", "difference"]:
            new_cells = this_cells[~mask]
            new_faces = this_faces[~mask]
        else:
            raise ValueError(f"Bad operation: {type}")

        if len(new_faces) > 0:
            new_faces = ravel_faces_to_vtk(new_faces)

        return SideSet(self.primary_mesh, new_cells, new_faces, name=set_name)

    def join(self, side_set, set_name: str = None):
        """
        Adds all faces within ``side_set`` to this object.

        Args
        ----
            side_set (SideSet): The side set to join.

        Returns
        -------
            joined_set: The joined SideSet.
        """
        return self._ufuncs(side_set, type="join", set_name=set_name)

    def remove(self, side_set, set_name: str = None):
        """
        Removes all faces within ``side_set`` from this object.

        Args
        ----
            side_set (SideSet): The side set to remove.

        Returns
        -------
            filtered_set: The filtered SideSet.
        """
        return self._ufuncs(side_set, type="remove", set_name=set_name)

    def intersection(self, side_set, set_name: str = None):
        """
        Returns a SideSet containing the intersection of this
        set and that of ``side_set``.

        Args
        ----
            side_set (SideSet): The side set to intersect.

        Returns
        -------
            intersected_set: The intersected SideSet.
        """
        return self._ufuncs(side_set, type="intersection", set_name=set_name)


class PointSet(object):
    def __init__(
        self,
        primary_mesh,
        primary_mesh_nodes,
        name: str = None,
        set_id: int = None,
    ):
        """
        Object that represents a collection of points from
        a mesh.
        """
        self.name = name
        self.primary_mesh = primary_mesh
        self.primary_nodes = primary_mesh_nodes
        self.set_id = set_id

        assert len(self.primary_nodes) > 0, "Set cannot be empty"

    @property
    def id(self):
        return self.set_id

    @id.setter
    def id(self, value):
        assert isinstance(value, int)
        self.set_id = value

    def __repr__(self):
        return f"PointSet<name={self.name}, num_points={len(self.primary_nodes)}>"

    def save(self, outfile: str):
        """
        Converts the set to a mesh and saves to ``outfile``.
        The extension for ``outfile`` can be any that Meshio supports:
            - ``.vtk``
            - ``.exo``
            - ``.inp``

        Args
        ----
            outfile (str): Path to save set to.
        """
        self.to_vtk_mesh().save(outfile)

    def to_vtk_mesh(self):
        """Returns the set as a VTK/PyVista mesh."""
        pts = self.primary_mesh.points[self.primary_nodes]
        return pv.PolyData(pts)

    @property
    def exodusii_format(self):
        """
        Returns set as a dictionary in an ExodusII-compliant schema.
        """
        return {
            "set_name": self.name,
            "set_id": self.set_id,
            "set_nodes": self.primary_nodes + 1,
            "num_set_nodes": len(self.primary_nodes),
            "num_set_dist_facts": 0,
        }

    def view(self, **kwargs):
        """
        Renders a 3D visualization of the set.

        Args
        ----
            **kwargs: Keyword arguments to pass to PyVista.
        """
        self.to_vtk_mesh().plot(**kwargs)

    def _ufuncs(self, point_set, type: str, set_name: str = None, set_id: int = None):
        """
        Master function for performing set operations.
        """
        assert (
            self.primary_mesh == point_set.primary_mesh
        ), "Sets must have the same parent mesh"

        if set_name is None:
            set_name = self.name

        if set_id is None:
            set_id = self.set_id

        this = self.primary_nodes
        that = point_set.primary_nodes

        if type in ["join", "union"]:
            new_points = np.union1d(this, that)
        elif type in ["intersect", "intersection"]:
            new_points = np.intersect1d(this, that)
        elif type in ["remove", "filter", "diff", "difference"]:
            new_points = np.setdiff1d(this, that)
        else:
            raise ValueError(f"Bad operation: {type}")

        return PointSet(self.primary_mesh, new_points, name=set_name, set_id=set_id)

    def join(self, point_set, set_name: str = None):
        """
        Adds all points within ``point_set`` to this object.

        Args
        ----
            point_set (PointSet): The point set to join.

        Returns
        -------
            joined_set: The joined PointSet.
        """
        return self._ufuncs(point_set, type="join", set_name=set_name)

    def remove(self, point_set, set_name: str = None):
        """
        Removes all points within ``point_set`` from this object.

        Args
        ----
            point_set (PointSet): The point set to remove.

        Returns
        -------
            filtered_set: The filtered PointSet.
        """
        return self._ufuncs(point_set, type="remove", set_name=set_name)

    def intersection(self, point_set, set_name: str = None):
        """
        Returns a PointSet containing the intersection of this
        set and that of ``point_set``.

        Args
        ----
            point_set (PointSet): The point set to intersect.

        Returns
        -------
            intersected_set: The intersected PointSet.
        """
        return self._ufuncs(point_set, type="intersection", set_name=set_name)


class ElementSet(object):
    def __init__(self, primary_mesh, primary_mesh_elements, name: str = None):
        self.name = name
        self.primary_mesh = primary_mesh
        self.primary_elements = primary_mesh_elements

    def __repr__(self):
        return f"ElementSet<name={self.name}, num_elements={len(self.primary_nodes)}>"

    def save(self, outfile: str):
        self.to_vtk_mesh().save(outfile)

    def to_vtk_mesh(self):
        pts = self.primary_mesh.points[self.primary_nodes]
        return pv.PolyData(pts)

    def view(self, **kwargs):
        """
        Renders a 3D visualization of the set.

        Args
        ----
            **kwargs: Keyword arguments to pass to PyVista.
        """
        self.to_vtk_mesh().plot(**kwargs)

    def join(self, element_set):
        """
        Adds all elements within ``side_set`` to this object.

        Args
        ----
            element_set (ElementSet): The element set to join.

        Returns
        -------
            joined_set: The joined ElementSet.
        """
        raise NotImplementedError()

    def remove(self, element_set):
        """
        Removes all faces within ``element_set`` from this object.

        Args
        ----
            element_set (ElementSet): The element set to remove.

        Returns
        -------
            filtered_set: The filtered ElementSet.
        """
        raise NotImplementedError()

    def intersection(self, element_set):
        """
        Returns a ElementSet containing the intersection of this
        set and that of ``element_set``.

        Args
        ----
            element_set (ElementSet): The element set to intersect.

        Returns
        -------
            intersected_set: The intersected ElementSet.
        """
        raise NotImplementedError()
