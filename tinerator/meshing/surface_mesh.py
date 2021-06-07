import numpy as np
import pyvista as pv
from scipy.spatial import distance
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

    @property
    def all_faces(self):
        return self.top_faces.join(self.bottom_faces).join(
            self.side_faces, set_name="AllFaces"
        )

    def get_layer(self, layer: tuple, return_faces: bool = True, set_name: str = None):

        layer_id_att = "cell_layer_id" if return_faces else "node_layer_id"
        arr = self._mesh.get_array(layer_id_att)

        if isinstance(layer, int):
            arr = np.floor(arr).astype(int)
        elif isinstance(layer, float):
            pass
        elif isinstance(layer, (list, tuple, np.ndarray)):
            layer = float(f"{layer[0]}.{layer[1]}")
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
        geom: np.array,
        close_ends: bool = True,
        at_layer: tuple = None,
        return_faces: bool = True,
        return_points: bool = False,
    ):
        """
        Coordinates can be either a list of (x,y) coordinates,
        or a Geometry object with points or lines.
        """

        side_faces_set = self.get_layer(1, 1)

        faces = self.faces
        node_map = self.node_mapping
        cell_map = self.cell_mapping

        side_faces = unravel_vtk_faces(side_faces_set.primary_faces)
        face_centroids = side_faces_set.to_vtk_mesh().cell_centers().points[:, :2]
        set_centroid = side_faces_set.to_vtk_mesh().center

        sort_origin = set_centroid[:2]

        lines = []
        if is_geometry(geom):
            coords = geom.coordinates  # TODO: handle differently for lines
            if close_ends:
                coords = np.vstack([coords, [coords[0]]])
            coords = [
                [60.9, 13.93],
                [77.9, 9.38],
                [69.4, 2.54],
                [17.5, 0.18],
                [2.7, 8.71],
                [18.2, 13.62],
            ]
            lines = [[coords[i], coords[i + 1]] for i in range(len(coords) - 1)]

        angles = np.array(
            [
                clockwiseangle_and_distance(pt, origin=sort_origin)
                for pt in face_centroids
            ]
        )
        sort_idx = np.lexsort((angles[:, 1], angles[:, 0]))

        sorted_face_centroids = face_centroids[sort_idx]

        sets = []

        for (i, line) in enumerate(np.array(lines)):
            print(line)
            face0, face1 = distance.cdist(line, sorted_face_centroids).argmin(axis=1)
            face0, face1 = np.sort([face0, face1 + 1])

            if face0 == face1:
                continue

            idx = sort_idx[np.array(range(face0, face1))]
            print(idx)

            import ipdb

            ipdb.set_trace()

            captured_cells = cell_map[idx]
            captured_surf_faces = faces[idx]
            captured_faces = np.hstack(
                [[len(fc), *node_map[fc.astype(int)]] for fc in captured_surf_faces]
            )

            sets.append(
                SideSet(
                    self.parent_mesh, captured_cells, captured_faces, name=f"Sides{i+1}"
                )
            )

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
