import numpy as np
import tempfile
import meshio
import pyvista as pv
from ..logging import log, warn, debug, _pylagrit_verbosity

def is_geometry(obj):
    try:
        return obj.__module__ == 'tinerator.gis.geometry' and type(obj).__name__ == 'Geometry'
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

    def join(self, side_set):
        """
        Adds all faces within ``side_set`` to this object.

        Args
        ----
            side_set (SideSet): The side set to join.
        
        Returns
        -------
            joined_set: The joined SideSet.
        """
        raise NotImplementedError()
    
    def remove(self, side_set):
        """
        Removes all faces within ``side_set`` from this object.

        Args
        ----
            side_set (SideSet): The side set to remove.
        
        Returns
        -------
            filtered_set: The filtered SideSet.
        """
        raise NotImplementedError()
    
    def intersection(self, side_set):
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
        raise NotImplementedError()



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

    def join(self, side_set):
        """
        Adds all points within ``side_set`` to this object.

        Args
        ----
            point_set (PointSet): The point set to join.
        
        Returns
        -------
            joined_set: The joined PointSet.
        """
        raise NotImplementedError()
    
    def remove(self, side_set):
        """
        Removes all points within ``point_set`` from this object.

        Args
        ----
            point_set (PointSet): The point set to remove.
        
        Returns
        -------
            filtered_set: The filtered PointSet.
        """
        raise NotImplementedError()
    
    def intersection(self, point_set):
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
        raise NotImplementedError()

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
        faces = []
        faces_vtk = self._mesh.faces

        i = 0
        sz_faces = len(faces_vtk)
        while True:
            if i >= sz_faces:
                break
            stride = faces_vtk[i]
            j = i + stride + 1
            faces.append(faces_vtk[i + 1 : j])
            i = j

        return np.array(faces, dtype=object)

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
        import ipdb; ipdb.set_trace()

        if is_geometry(coordinates):
            coordinates = coordinates.coordinates

        from scipy.spatial import distance

        # where cell_layer_id == (1,1)
        # where sides = true
        # intersection(sides, cell_layer_id)

        #top_nodes = self.top_nodes
        #primary_mesh = top_nodes.primary_mesh
        #primary_nodes = top_nodes.primary_nodes
        #boundary = 

        #boundary

        # - get boundary nodes from top 

        #facesets = {}

        #for key in coords:
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

        #return facesets

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
