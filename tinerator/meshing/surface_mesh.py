import os
import numpy as np
import tempfile
import meshio
import pyvista as pv
from ..logging import log, warn, debug, _pylagrit_verbosity

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
            mesh_name = f"(\"{mesh_obj.name}\")" if mesh_obj.name is not None else ""
            mesh_name = f"Point Set {mesh_name}".strip()
            mesh_obj = mesh_obj.to_vtk_mesh()
            kwargs['color'] = 'red'
            kwargs['render_points_as_spheres'] = True
        elif isinstance(mesh_obj, SideSet):
            mesh_name = f"(\"{mesh_obj.name}\")" if mesh_obj.name is not None else ""
            mesh_name = f"Side Set {mesh_name}".strip()
            mesh_obj = mesh_obj.to_vtk_mesh()
        else:
            mesh_name = "Primary Mesh"
            kwargs['show_edges'] = True

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

class PointSet(object):
    def __init__(self, primary_mesh, primary_mesh_nodes, surface_mesh = None, surface_mesh_nodes: int = None, name: str = None):
        self.name = name
        self.primary_mesh = primary_mesh
        self.primary_nodes = primary_mesh_nodes
        self.surface_mesh = surface_mesh
        self.surface_nodes = surface_mesh_nodes
    
    def save(self, outfile: str):
        self.to_vtk_mesh().save(outfile)
    
    def to_vtk_mesh(self):
        pts = self.primary_mesh.points[self.primary_nodes]
        return pv.PolyData(pts)

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
        self._mesh = mesh.extract_surface(pass_pointid=True, pass_cellid=True, nonlinear_subdivision=0)

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
        return self._mesh.cells

    #@property
    #def node_id(self):
    #    """The original node IDs of the parent mesh."""
    #    return self._mesh.point_data["idnode0"].astype(int)

    #@property
    #def cell_id(self):
    #    """The original cell IDs of the parent mesh."""
    #    return np.hstack(self._mesh.cell_data["idelem1"]).astype(int)

    #@property
    #def material_id(self):
    #    """Returns the equivalent material ID of the parent mesh."""
    #    return np.hstack(self._mesh.cell_data["itetclr1"])

    #@property
    #def face_id(self):
    #    """The original face IDs of the parent mesh."""
    #    return np.hstack(self._mesh.cell_data["idface1"]).astype(int)

    #@property
    #def top_faces(self):
    #    """Returns a SideSet object of all top-layer faces."""
    #    eset = np.argwhere(self._layertyp_elem == SurfaceMesh.LAYER_TOP).T[0]
    #    return SideSet("TopFaces", self.cell_id[eset], self.face_id[eset])

    #@property
    #def top_points(self):
    #    """Returns a NodeSet object of all top-layer points."""
    #    pset = np.argwhere(self._layertyp == SurfaceMesh.LAYER_TOP).T[0]
    #    return NodeSet("TopNodes", self.node_id[pset])

    #@property
    #def bottom_faces(self):
    #    """Returns a SideSet object of all bottom-layer faces."""
    #    eset = np.argwhere(self._layertyp_elem == SurfaceMesh.LAYER_BOTTOM).T[0]
    #    return SideSet("BottomFaces", self.cell_id[eset], self.face_id[eset])

    #@property
    #def side_faces(self):
    #    """Returns a SideSet object of all side-layer faces."""
    #    eset = np.argwhere(self._layertyp_elem == SurfaceMesh.LAYER_SIDES).T[0]
    #    return SideSet("SideFaces", self.cell_id[eset], self.face_id[eset])

    #@property
    #def top_faces(self):
    #    return get_faces_where(mesh, surf_mesh, attribute_name="layertyp_cell", attribute_value=-2)

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
            faces.append(faces_vtk[i+1:j])
            i = j
        
        return np.array(faces, dtype=object)

    @property
    def cell_mapping(self):
        return self._mesh.get_array('vtkOriginalCellIds')
    
    @property
    def node_mapping(self):
        return self._mesh.get_array('vtkOriginalPointIds')

    def get_attribute(self, attribute_name: str):
        return self._mesh.get_array(attribute_name)

    def get_nodes_where(self, attribute_name: str = None, attribute_value = None, set_name: str = None):
        primary_att = self.parent_mesh.get_array(attribute_name)
        surf_att = self._mesh.get_array(attribute_name)

        primary_nodes = np.argwhere(primary_att == attribute_value).T[0]
        surf_nodes = np.argwhere(surf_att == attribute_value).T[0]

        return PointSet(
            self.parent_mesh,
            primary_nodes,
            surface_mesh = self._mesh,
            surface_mesh_nodes = surf_nodes,
            name = set_name
        )

    @property
    def top_nodes(self):
        return self.get_nodes_where("layertyp", -2., set_name = "TopPoints")

    @property
    def bottom_nodes(self):
        return self.get_nodes_where("layertyp", -1., set_name = "BottomPoints")
    
    @property
    def side_nodes(self):
        return self.get_nodes_where("layertyp", 0., set_name = "SidePoints")
    
    def get_faces_where(self, attribute_name: str = None, attribute_value = None, set_name: str = None):

        faces = self.faces

        cell_map = self.cell_mapping
        node_map = self.node_mapping
        primary_att = self.parent_mesh.get_array(attribute_name)
        surf_att = self._mesh.get_array(attribute_name)

        primary_cells = np.argwhere(primary_att == attribute_value).T[0]
        face_idx = np.argwhere(surf_att == attribute_value).T[0]

        captured_cells = cell_map[face_idx]
        captured_surf_faces = faces[face_idx]
        captured_faces = np.hstack([[len(fc), *node_map[fc]] for fc in captured_surf_faces])

        return SideSet(
            self.parent_mesh,
            captured_cells,
            captured_faces,
            name = set_name,
        )

    @property
    def side_faces(self):
        return self.get_faces_where("layertyp_cell", 0., set_name = "SideFaces")

    @property
    def bottom_faces(self):
        return self.get_faces_where("layertyp_cell", -1., set_name = "BottomFaces")

    @property
    def top_faces(self):
        return self.get_faces_where("layertyp_cell", -2., set_name = "TopFaces")