import numpy as np
import tempfile
import meshio
import pyvista as pv
from ..logging import log, warn, debug, _pylagrit_verbosity


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

        # vtk_face = np.hstack([np.count_nonzero(~np.isnan(face)), face])
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

    a = as_void(a)
    b = as_void(b)

    return np.in1d(a, b, assume_unique)


def is_geometry(obj):
    try:
        return (
            obj.__module__ == "tinerator.gis.geometry"
            and type(obj).__name__ == "Geometry"
        )
    except:
        return False


def plot_sets(mesh, sets, num_cols: int = 3, link_views: bool = True):
    """
    Plots the mesh along with element sets, side (face) sets, and
    point sets.
    """
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

        if isinstance(mesh_obj, PointSet):
            mesh_name = f'("{mesh_obj.name}")' if mesh_obj.name is not None else ""
            mesh_name = f"Point Set {mesh_name}".strip()
            mesh_obj = mesh_obj.to_vtk_mesh()
            kwargs["color"] = "red"
            kwargs["render_points_as_spheres"] = True
        elif isinstance(mesh_obj, SideSet):
            mesh_name = f'("{mesh_obj.name}")' if mesh_obj.name is not None else ""
            mesh_name = f"Side Set {mesh_name}".strip()
            mesh_obj = mesh_obj.to_vtk_mesh()
        else:
            mesh_name = "Primary Mesh"
            kwargs["show_edges"] = True

        p.add_text(mesh_name, font_size=12)
        p.add_mesh(mesh_obj, **kwargs)

    if link_views:
        p.link_views()

    p.show()


class SideSet(object):
    def __init__(self, primary_mesh, primary_cells, primary_faces, name: str = None):
        """
        ``primary_faces`` should take the form of:

            (primary_element_id, primary_element_face_id)
        """
        self.name = name
        self.primary_mesh = primary_mesh
        self.primary_cells = primary_cells
        self.primary_faces = primary_faces

    def save(self, outfile: str):
        self.to_vtk_mesh().save(outfile)

    def to_vtk_mesh(self):
        pts = self.primary_mesh.points
        faces = self.primary_faces
        return pv.PolyData(pts, faces)

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
            new_cells = np.hstack([this_cells[~mask], that_cells])
            new_faces = np.vstack([this_faces[~mask], that_faces])
        elif type in ["intersect", "intersection"]:
            new_cells = this_cells[mask]
            new_faces = this_faces[mask]
        elif type in ["remove", "filter", "diff", "difference"]:
            new_cells = that_faces[~mask]
            new_faces = this_faces[~mask]
        else:
            raise ValueError(f"Bad operation: {type}")

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
    ):
        self.name = name
        self.primary_mesh = primary_mesh
        self.primary_nodes = primary_mesh_nodes

        assert len(self.primary_nodes) > 0, "Set cannot be empty"

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

    def _ufuncs(self, point_set, type: str, set_name: str = None):
        """
        Master function for performing set operations.
        """
        assert (
            self.primary_mesh == point_set.primary_mesh
        ), "Sets must have the same parent mesh"

        if set_name is None:
            set_name = self.name

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

        return PointSet(self.primary_mesh, new_points, name=set_name)

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


class SurfaceMesh:

    LAYER_TOP = -2
    LAYER_BOTTOM = -1
    LAYER_SIDES = 0

    # The corresponding direction to the faces:
    # https://lagrit.lanl.gov/docs/supported.html
    PRISM_FACE_TOP = 2
    PRISM_FACE_BOTTOM = 1
    PRISM_FACE_LEFT = 3
    PRISM_FACE_RIGHT = 5
    PRISM_FACE_BACK = 4

    def __init__(self, mesh):
        """Extracts the surface mesh from a volume mesh."""
        self.parent_mesh = mesh
        self._mesh = mesh.extract_surface(
            pass_pointid=True, pass_cellid=True, nonlinear_subdivision=0
        )

    def save(self, outfile: str, **kwargs):
        """Writes the mesh to disk. Uses Meshio as a file writer."""
        self._mesh.save(outfile, **kwargs)

    def view(self, **kwargs):
        self._mesh.plot(**kwargs)

    def __repr__(self):
        return f"SurfaceMesh<nodes={len(self.nodes)}, cells={len(self.cells)}>"

    @property
    def nodes(self):
        return self._mesh.points

    @property
    def cells(self):
        return self.faces

    @property
    def faces(self):
        return unravel_vtk_faces(self.faces_vtk)

    @property
    def faces_vtk(self):
        return self._mesh.faces

    @property
    def cell_mapping(self):
        return self._mesh.get_array("vtkOriginalCellIds")

    @property
    def node_mapping(self):
        return self._mesh.get_array("vtkOriginalPointIds")

    def point_data_to_cell_data(self, attribute_name: str, target_attribute_name: str):
        from pyvista import _vtk

        alg = _vtk.vtkPointDataToCellData()
        alg.SetInputDataObject(self._mesh)
        alg.SetPassPointData(True)
        alg.CategoricalDataOn()
        alg.Update()

        mesh_new = pv.core.filters._get_output(alg, active_scalars=None)
        cell_array = mesh_new.get_array(attribute_name, preference="cell")

        self._mesh[target_attribute_name] = cell_array

    def get_attribute(self, attribute_name: str):
        return self._mesh.get_array(attribute_name)

    def get_nodes_where(
        self, attribute_name: str = None, attribute_value=None, set_name: str = None
    ):
        node_map = self.node_mapping
        surf_att = self._mesh.get_array(attribute_name)
        surf_nodes = np.argwhere(surf_att == attribute_value).T[0]

        primary_nodes = node_map[surf_nodes]

        return PointSet(
            self.parent_mesh,
            primary_nodes,
            name=set_name,
        )

    @property
    def top_nodes(self):
        return self.get_nodes_where("layertyp", -2.0, set_name="TopPoints")

    @property
    def bottom_nodes(self):
        return self.get_nodes_where("layertyp", -1.0, set_name="BottomPoints")

    @property
    def side_nodes(self):
        return self.get_nodes_where("layertyp", 0.0, set_name="SidePoints")

    def get_faces_where(
        self, attribute_name: str = None, attribute_value=None, set_name: str = None
    ):

        faces = self.faces

        cell_map = self.cell_mapping
        node_map = self.node_mapping
        surf_att = self._mesh.get_array(attribute_name)

        face_idx = np.argwhere(surf_att == attribute_value).T[0]

        captured_cells = cell_map[face_idx]
        captured_surf_faces = faces[face_idx]
        captured_faces = np.hstack(
            [[len(fc), *node_map[fc]] for fc in captured_surf_faces]
        )

        return SideSet(
            self.parent_mesh,
            captured_cells,
            captured_faces,
            name=set_name,
        )

    @property
    def side_faces(self):
        return self.get_faces_where("layertyp_cell", 0.0, set_name="SideFaces")

    @property
    def bottom_faces(self):
        return self.get_faces_where("layertyp_cell", -1.0, set_name="BottomFaces")

    @property
    def top_faces(self):
        return self.get_faces_where("layertyp_cell", -2.0, set_name="TopFaces")

    def get_layer(self, layer: tuple, return_faces: bool = True, set_name: str = None):

        layer_id_att = "cell_layer_id" if return_faces else "node_layer_id"
        arr = self._mesh.get_array(layer_id_att)

        if isinstance(layer, int):
            arr = np.floor(arr).astype(int)
        elif isinstance(layer, float):
            pass
            # idx = np.argwhere(self._mesh.get_array(layer_id_att) == layer).T[0]
        elif isinstance(layer, (list, tuple, np.ndarray)):
            layer = float(f"{layer[0]}.{layer[1]}")
            # idx = np.argwhere(self._mesh.get_array(layer_id_att) == val).T[0]
        else:
            raise ValueError("Could not parse layer")

        idx = np.argwhere(arr == layer).T[0]
        node_map = self.node_mapping

        if return_faces:
            faces = self.faces
            cell_map = self.cell_mapping

            captured_cells = cell_map[idx]
            captured_surf_faces = faces[idx]
            captured_faces = np.hstack(
                [[len(fc), *node_map[fc]] for fc in captured_surf_faces]
            )

            ss = SideSet(
                self.parent_mesh, captured_cells, captured_faces, name=set_name
            )
            return ss.remove(self.top_faces).remove(self.bottom_faces)
        else:
            primary_nodes = node_map[idx]

            ps = PointSet(
                self.parent_mesh,
                primary_nodes,
                name=set_name,
            )
            return ps.remove(self.top_nodes).remove(self.bottom_nodes)

    def discretize_top(
        self, heights: np.array, return_faces: bool = True, return_points: bool = False
    ):
        """
        ``heights`` can either be a NumPy array of elevations,
        or a Geometry object composed of polygons.
        """
        raise NotImplementedError()

    def discretize_sides(
        self,
        coordinates: np.array,
        close_ends: bool = True,
        at_layer: tuple = None,
        return_faces: bool = True,
        return_points: bool = False,
    ):
        """
        Coordinates can be either a list of (x,y) coordinates,
        or a Geometry object with points or lines.
        """
        import ipdb

        ipdb.set_trace()

        if is_geometry(coordinates):
            coordinates = coordinates.coordinates  # TODO: handle differently for lines

        from scipy.spatial import distance

        side_faces = self.side_faces

        # if at_layer is not None:

        # where cell_layer_id == (1,1)
        # where sides = true
        # intersection(sides, cell_layer_id)

        # top_nodes = self.top_nodes
        # primary_mesh = top_nodes.primary_mesh
        # primary_nodes = top_nodes.primary_nodes
        # boundary =

        # boundary

        # - get boundary nodes from top

        # facesets = {}

        # for key in coords:
        #    mat_ids = np.full((np.shape(boundary)[0],),1,dtype=int)
        #    fs = []

        #    # Iterate over given coordinates and find the closest boundary point...
        #    for c in coords[key]:
        #        ind = distance.cdist([c], boundary[:,:2]).argmin()
        #        fs.append(ind)

        #    # TODO: Band-aid fix.
        #    if len(fs) == 2:
        #        # Reverse order of fs, only so that the
        #        # notion of 'clockwise ordering matters'
        #        # stays constant.
        #        fs = fs[::-1]
        #    else:
        #        fs.sort(reverse=True)

        #    # Map the interim space as a new faceset.
        #    # 'Unmarked' facesets have a default filled value of 1
        #    for i in range(len(fs)):

        #        if fs[-1] > fs[i]:
        #            mat_ids[fs[-1]:] = i+2
        #            mat_ids[:fs[i]] = i+2
        #        else:
        #            mat_ids[fs[-1]:fs[i]] = i+2

        #    facesets[key] = mat_ids

        # return facesets

    def from_geometry(self, geometry):
        """
        Takes in a TINerator Geometry object and returns
        a Side Set or a Point Set where the mesh intersects.
        """
        raise NotImplementedError()
        # - rasterize geometry
        # - map to attribute
        # - map to layer (1,1)
        # - return where faces intersect
        # - optionally, return points where:
        #       - get faces intersect
        #       - get closest node in face
