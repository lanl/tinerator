import numpy as np
import pyvista as pv
from scipy.spatial import distance
from shapely.geometry import Polygon
from ..logging import log, warn, debug, _pylagrit_verbosity
from .mesh_metrics import get_cell_normals
from .meshing_utils import (
    clockwiseangle_and_distance,
    ravel_faces_to_vtk,
    unravel_vtk_faces,
    refit_arrays,
    in2d,
    is_geometry,
)
from .sets import SideSet, PointSet, ElementSet
from .adjacency import UndirectedGraph


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

    def view(self, *args, **kwargs):
        """Visualize the surface mesh."""
        self._mesh.plot(*args, **kwargs)

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

        assert isinstance(attribute_name, str)

        faces = self.faces

        cell_map = self.cell_mapping
        node_map = self.node_mapping
        surf_att = self._mesh.get_array(attribute_name)

        face_idx = np.argwhere(surf_att == attribute_value).T[0]

        captured_cells = cell_map[face_idx]
        captured_surf_faces = faces[face_idx]
        captured_faces = np.hstack(
            [[len(fc), *node_map[fc.astype(int)]] for fc in captured_surf_faces]
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

    def discretize_top(self, attribute_name: str):
        """
        Returns a side set for each unique value in an attribute.
        Attribute *must be* discrete.

        Args:
            attribute_name (str): The attribute name to query from.

        Returns:
            List[SideSet]: A list of side sets.
        """
        bottom_faces = self.bottom_faces
        side_faces = self.side_faces

        attribute_values = np.unique(self._mesh.get_array(attribute_name))

        if len(attribute_values) > 100:
            warn(
                f"{len(attribute_values)} side sets will be created. "
                "This function should only be used for discrete attributes."
            )

        if np.allclose(attribute_values, attribute_values.astype(int)):
            attribute_values = attribute_values.astype(int)

        sets = []
        for value in attribute_values:
            new_set = self.get_faces_where(
                attribute_name=attribute_name, attribute_value=value
            )
            new_set.name = f"{attribute_name}={value}"
            sets.append(new_set.remove(side_faces).remove(bottom_faces))

        return sets

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
        for (sublayer_id, layer) in enumerate(layers):
            log(f"Generating side set for sublayer {sublayer_id+1} of {len(layers)}")

            layer_vtk = layer.to_vtk_mesh()
            min_x, max_x, min_y, max_y, _, _ = layer_vtk.bounds

            layer_centroids = np.array(layer_vtk.cell_centers().points)[:, :2]
            layer_center = np.array(layer_vtk.center)[:2]
            layer_faces = unravel_vtk_faces(layer.primary_faces, fill_matrix=True)

            adj_graph = UndirectedGraph(layer_vtk)
            num_layer_cells = layer_centroids.shape[0]
            visited = np.zeros((num_layer_cells,), dtype=int)

            # Find the leftmost, center cell to use as the first iteration.
            start_pt = [min_x, (max_y - min_y) / 2.0 + min_y]
            current_cell_id = distance.cdist([start_pt], layer_centroids).argmin(
                axis=1
            )[0]

            # Here, we are traversing the undirected graph in order to get the correct side
            # element ordering. We start at the middle-left cell and traverse the graph
            # in a clockwise fashion.
            iter_visited = 0
            while not visited.all():
                visited[current_cell_id] = iter_visited
                adj_cell_ids = np.array(adj_graph.get_adjacent_nodes(current_cell_id))
                adj_cell_ids = adj_cell_ids[visited[adj_cell_ids] == 0]

                if len(adj_cell_ids) == 1:
                    current_cell_id = adj_cell_ids[0]
                elif len(adj_cell_ids) > 1:
                    # Assuming a connected graph, this should only be called once:
                    # when trying to decide which direction to travel on the first iteration
                    angles = np.array(
                        [
                            clockwiseangle_and_distance(pt, origin=layer_center)
                            for pt in layer_centroids[adj_cell_ids]
                        ]
                    )
                    current_cell_id = adj_cell_ids[
                        np.lexsort((angles[:, 0], angles[:, 1]))[-1]
                    ]
                else:
                    debug("Reached end of adjacency graph.")
                    debug(f"Have all points been visited? {visited.all()}")

                    if not visited.all():
                        warn(
                            "Side set traversal seems to have failed. Are there unconnected cells?"
                        )

                    break

                iter_visited += 1

            sort_idx = np.argsort(visited)
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

    def from_geometry(self, geometry, set_name: str = "Geometry"):
        """
        From a TINerator Geometry object, returns a set of all top-surface
        faces that intersect with the Geometry object.

        As an example, this can be used to capture flowline cells generated via
        watershed delineation.

        Args
        ----
            geometry (tinerator.gis.Geometry): The geometry object to check for intersections.
            set_name (:obj:`str`, optional): The name for the set.

        Returns
        -------
            side_set

        Examples
        --------
            >>> ws_delin = tin.gis.watershed_delineation(dem)
            >>> side_set = surface_mesh.from_geometry(ws_delin)
        """

        # We only care about the top faces - get those
        top_faces = self.top_faces
        vtk_top_faces = self.top_faces.to_vtk_mesh()

        # Reform the top faces from an index into the points array
        # into actually storing the points
        points = vtk_top_faces.points
        faces = unravel_vtk_faces(vtk_top_faces.faces)
        faces_with_pts = [np.array(points[face.astype(int)]) for face in faces]

        # Convert all top faces into Shapely Polygon objects
        faces_as_polygons = []
        for (id, face) in enumerate(faces_with_pts):
            polygon = Polygon(face[:, :2].tolist() + [face[0][:2].tolist()])
            faces_as_polygons.append((id, polygon))

        # Finally, run the Shapely `intersects` object between
        # each pair of top faces and Geometry shapes
        # If there is an intersection, then it's marked for the new set
        mask = np.zeros((len(faces_as_polygons),), dtype=bool)
        for (i, triangle) in faces_as_polygons:
            for shape in geometry.shapes:
                if triangle.intersects(shape):
                    mask[i] = True

        primary_cells = top_faces.primary_cells[mask]
        primary_faces = ravel_faces_to_vtk(faces[mask].astype(int))

        return SideSet(
            top_faces.primary_mesh,
            primary_cells,
            primary_faces,
            name=set_name,
            set_id=None,
        )

    def from_cell_normals(
        self,
        get_north: bool = True,
        get_south: bool = True,
        get_west: bool = True,
        get_east: bool = True,
        get_top: bool = True,
        get_bottom: bool = False,
        tol: float = 0.03,
    ):
        compass = {
            "north": [0, +1, 0],
            "south": [0, -1, 0],
            "east": [+1, 0, 0],
            "west": [-1, 0, 0],
            "up": [0, 0, +1],
            "down": [0, 0, -1],
        }

        normals = get_cell_normals(self._mesh)

        node_map = self.node_mapping
        faces = self.faces
        cell_map = self.cell_mapping

        sets = []

        for compass_dir in compass:
            v = compass[compass_dir]
            diff = np.mean(np.abs(normals - np.array(v, dtype=float)), axis=1)
            idx = np.argwhere(diff < tol).T[0]

            if idx.size == 0:
                continue

            captured_cells = cell_map[idx]
            captured_surf_faces = faces[idx]
            captured_faces = np.hstack(
                [[len(fc), *node_map[fc.astype(int)]] for fc in captured_surf_faces]
            )

            set = SideSet(
                self.parent_mesh, captured_cells, captured_faces, name=compass_dir
            )
            sets.append(set)

        return sets

    def validate_sets(self, sets):
        """
        Given a list of side sets, this function verifies their integrity.
        Namely, it checks that they are non-overlapping, and that they cover
        the whole of the mesh surface.

        Raises an AssertionError if malformed.

        Args:
            sets (List[SideSet]): A list of sets to check.
        """
        all_faces = self.all_faces

        union_test = sets[0]
        for s in sets[1:]:
            union_test = union_test.join(s)

        assert len(all_faces.primary_cells) == len(
            union_test.primary_cells
        ), "Malformed set(s): cell count mismatch"
        assert len(all_faces.primary_faces) == len(
            union_test.primary_faces
        ), "Malformed set(s): face count mismatch"

        log("[green]PASSED[/green]: union")

        difference_test = union_test.remove(all_faces)

        assert (
            len(difference_test.primary_cells) == 0
        ), "Malformed set(s): cell count mismatch"
        assert (
            len(difference_test.primary_faces) == 0
        ), "Malformed set(s): face count mismatch"

        log("[green]PASSED[/green]: difference")

        intersection_test = []

        for i in range(len(sets)):
            for j in range(len(sets)):
                if i == j:
                    continue

                intersection_test.append(sets[i].intersection(sets[j]))
                intersection_test[-1].name = f"{sets[i].name} | {sets[j].name}"

        for intersection in intersection_test:
            assert (
                len(intersection.primary_cells) == 0
            ), f"Malformed set(s): cell count mismatch ({intersection.name})"
            assert (
                len(intersection.primary_faces) == 0
            ), f"Malformed set(s): face count mismatch ({intersection.name})"

        log("[green]PASSED[/green]: intersection")
