import numpy as np
import pyvista as pv
from scipy.spatial import distance
from ..constants import _in_notebook, JUPYTER_BACKEND_DEFAULT
from ..logging import log, warn, debug, _pylagrit_verbosity
from .meshing_utils import (
    clockwiseangle_and_distance,
    ravel_faces_to_vtk,
    unravel_vtk_faces,
    refit_arrays,
    in2d,
    is_geometry,
)
from .sets import SideSet, PointSet, ElementSet


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
        """Visualize the surface mesh."""
        if _in_notebook():
            jupyter_backend = JUPYTER_BACKEND_DEFAULT
        else:
            jupyter_backend = None
        self._mesh.plot(jupyter_backend=jupyter_backend, **kwargs)

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
        """Returns an attribute as a NumPy array."""
        return self._mesh.get_array(attribute_name)

    def get_nodes_where(
        self, attribute_name: str = None, attribute_value=None, set_name: str = None
    ):
        """
        Gets external points where ``attribute_name`` has the value ``attribute_value``.

        Args
        ----
            attribute_name (str): The attribute to query.
            attribute_value (int, float): The attribute value to get.
            set_name (:obj:`str`, optional): The desired set name.

        Returns
        -------
            set : a Point Set object.

        Examples
        --------
            >>> set = surf_mesh.get_faces_where(attribute_name="node_color", attribute_value=1)
        """
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
        """Returns all top nodes."""
        return self.get_nodes_where("layertyp", -2.0, set_name="TopPoints")

    @property
    def bottom_nodes(self):
        """Returns all bottom nodes."""
        return self.get_nodes_where("layertyp", -1.0, set_name="BottomPoints")

    @property
    def side_nodes(self):
        """Returns all side nodes."""
        return self.get_nodes_where("layertyp", 0.0, set_name="SidePoints")

    def get_faces_where(
        self, attribute_name: str = None, attribute_value=None, set_name: str = None
    ):
        """
        Gets external faces where ``attribute_name`` has the value ``attribute_value``.

        Args
        ----
            attribute_name (str): The attribute to query.
            attribute_value (int, float): The attribute value to get.
            set_name (:obj:`str`, optional): The desired set name.

        Returns
        -------
            set : a Side Set object.

        Examples
        --------
            >>> set = surf_mesh.get_faces_where(attribute_name="material_id", attribute_value=1)
        """

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
        """Returns all side faces."""
        return self.get_faces_where("layertyp_cell", 0.0, set_name="SideFaces")

    @property
    def bottom_faces(self):
        """Returns all bottom faces."""
        return self.get_faces_where("layertyp_cell", -1.0, set_name="BottomFaces")

    @property
    def top_faces(self):
        """Returns all top faces."""
        return self.get_faces_where("layertyp_cell", -2.0, set_name="TopFaces")

    @property
    def all_faces(self):
        """
        Returns all exterior faces.
        """
        return self.top_faces.join(self.bottom_faces).join(
            self.side_faces, set_name="AllFaces"
        )

    @property
    def layer_ids(self):
        """Returns the available layers in the mesh."""
        layers_float = np.unique(self._mesh.get_array("cell_layer_id"))
        return [tuple(map(int, str(x).split("."))) for x in layers_float]

    def get_layer(self, layer: tuple, return_faces: bool = True, set_name: str = None):
        """
        Returns a side set or a point set for all sides/points on the exterior of the mesh
        at a given layer.

        Args
        ----
            layer (tuple): The layer to get the set at.
            return_faces (:obj:`bool`, optional): If True, return a Side Set. Otherwise, return a Point Set.
            set_name (:obj:`str`, optional): The name to give the set.

        Returns
        -------
            set : a Side Set or Point Set.

        Examples
        --------
            >>> surface_mesh.get_layer((1,1)) # get the top layer
            >>> surface_mesh.get_layer(2, return_faces = False) # get all of layer 2 as a point set
        """

        layer_id_att = "cell_layer_id" if return_faces else "node_layer_id"
        arr = self._mesh.get_array(layer_id_att)

        if isinstance(layer, int):
            arr = np.floor(arr).astype(int)
        elif isinstance(layer, float):
            pass
        elif isinstance(layer, (list, tuple, np.ndarray)):
            layer = float(f"{layer[0]}.{layer[1]}")
        else:
            raise ValueError(f"Could not parse layer: {layer}")

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
        geom: np.array,
        close_ends: bool = True,
        at_layer: tuple = None,
        set_name_prefix: str = "Sides",
    ):
        """
        Discretizes the side external faces of the mesh with
        the given coordinates.

        Coordinates must be given in clockwise ordering.

        Args
        ----
            geom (:obj:`np.ndarray`): a list of (x, y) coordinates
            close_ends (:obj:`bool`, optional): If True, discretizes the entire perimeter
            at_layer (:obj:`tuple`, optional): Generate a set from a specified layer
            set_name_prefix (:obj:`str`, optional): The set name prefix

        Returns
        -------
            sets : a list of side sets.

        Examples
        --------
            >>> sides = [
                (65.5, 0.6),
                (50.5, 0.04),
                (12.0, 0.04),
                (12.0, 12.9),
                (79.1, 9.1),
            ]
            >>> outlet = [
                (71.9, 4.7),
                (67.3, 2.1),
            ]
            >>> sides_side_set = sf.discretize_sides(sides)
            >>> outlet = sf.discretize_sides(
                    outlet,
                    close_ends = False,
                    at_layer = (1, 1),
                    set_name_prefix="Outlet"
                )
        """

        if isinstance(geom, (list, tuple, np.ndarray)):
            geom = list(geom)
            if close_ends:
                geom += [geom[0]]
            geom = [[geom[i], geom[i + 1]] for i in range(len(geom) - 1)]
        else:
            raise NotImplementedError()

        # Take the sides from all layers if not specified
        if at_layer is not None:
            layers = [self.get_layer(at_layer)]
        else:
            layers = [self.get_layer(x) for x in self.layer_ids]

        sets = []

        # Iterate over every layer, compute the (sorted) boundary,
        # and grab faces between start and end points
        for layer in layers:
            layer_vtk = layer.to_vtk_mesh()
            layer_centroids = np.array(layer_vtk.cell_centers().points)[:, :2]
            layer_center = np.array(layer_vtk.center)[:2]

            layer_faces = unravel_vtk_faces(layer.primary_faces, fill_matrix=True)

            # Sort the boundary clockwise
            angles = np.array(
                [
                    clockwiseangle_and_distance(pt, origin=layer_center)
                    for pt in layer_centroids
                ]
            )
            sort_idx = np.lexsort((angles[:, 1], angles[:, 0]))
            layer_centroids_sorted = layer_centroids[sort_idx]

            layer_sets = []
            slices = []
            for (i, line) in enumerate(geom):
                start, end = distance.cdist(
                    np.array(line), layer_centroids_sorted
                ).argmin(axis=1)

                if start == end:
                    warn("Malformed line: start equals end")
                    continue

                if start > end:
                    slices.append(
                        np.hstack(
                            [
                                np.array(
                                    range(start, len(layer_centroids_sorted)), dtype=int
                                ),
                                np.array(range(0, end), dtype=int),
                            ]
                        )
                    )
                else:
                    slices.append(np.array(range(start, end), dtype=int))

            # Convert those sorted indices into an actual side set
            for idx in slices:
                captured_cells = layer.primary_cells[sort_idx[idx]]
                captured_faces = ravel_faces_to_vtk(layer_faces[sort_idx[idx]])
                layer_sets.append(
                    SideSet(self.parent_mesh, captured_cells, captured_faces)
                )

            # Join subsequent layers to the top layer
            if sets:
                sets = [sets[i].join(layer_sets[i]) for i in range(len(layer_sets))]
            else:
                sets = layer_sets

        if len(sets) == 1:
            set = sets[0]
            set.name = set_name_prefix
            return set

        for (i, set) in enumerate(sets):
            set.name = f"{set_name_prefix}={i+1}"

        return sets

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
