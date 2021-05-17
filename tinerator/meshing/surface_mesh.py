import os
import numpy as np
import tempfile
import meshio
from pylagrit import PyLaGriT
from copy import copy, deepcopy
from enum import Enum, auto
from .mesh import Mesh, StackedMesh, ElementType, load_mesh
from ..logging import log, warn, debug, _pylagrit_verbosity

# https://www.attrs.org/en/stable/overview.html
# https://github.com/ecoon/watershed-workflow/blob/c1b593e79e96b8fe22685d38c8777df0d824b1f6/workflow/mesh.py#L71
class SideSet(object):
    def __init__(self, name: str, elem_list: list, side_list: list, setid: int = None):
        self.name = name
        self.elem_list = elem_list
        self.side_list = side_list
        self.setid = setid

    def __repr__(self):
        return f'SideSet(name="{self.name}", setid={self.setid})'


class NodeSet(object):
    def __init__(self, name: str, nodes_list: int, setid: int = None):
        self.name = name
        self.nodes_list = nodes_list
        self.setid = setid

    def __repr__(self):
        return f'NodeSet(name="{self.name}", setid={self.setid})'


class ElemSet(object):
    def __init__(self, name):
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

    def __init__(self, volume_mesh: Mesh):
        """Extracts the surface mesh from a volume mesh."""
        self._mesh = extract_surface_mesh(volume_mesh)
        self._layertyp = self._mesh.point_data["layertyp"].astype(int)
        self._layertyp_elem = np.hstack(self._mesh.cell_data["layertypelem"]).astype(
            int
        )

    def save(self, outfile: str, file_format: str = None, **kwargs):
        """Writes the mesh to disk. Uses Meshio as a file writer."""
        if file_format is None:
            if os.path.splitext(outfile)[-1].lower() == ".inp":
                file_format = "avsucd"

        self._mesh.write(outfile, file_format=file_format)

    def __repr__(self):
        return f"SurfaceMesh<nodes={len(self.nodes)}, cells={len(self.cells)}>"

    @property
    def nodes(self):
        return self._mesh.points

    @property
    def cells(self):
        return self._mesh.cells

    @property
    def node_id(self):
        """The original node IDs of the parent mesh."""
        return self._mesh.point_data["idnode0"].astype(int)

    @property
    def cell_id(self):
        """The original cell IDs of the parent mesh."""
        return np.hstack(self._mesh.cell_data["idelem1"]).astype(int)

    @property
    def material_id(self):
        """Returns the equivalent material ID of the parent mesh."""
        return np.hstack(self._mesh.cell_data["itetclr1"])

    @property
    def face_id(self):
        """The original face IDs of the parent mesh."""
        return np.hstack(self._mesh.cell_data["idface1"]).astype(int)

    @property
    def top_faces(self):
        """Returns a SideSet object of all top-layer faces."""
        eset = np.argwhere(self._layertyp_elem == SurfaceMesh.LAYER_TOP).T[0]
        return SideSet("TopFaces", self.cell_id[eset], self.face_id[eset])

    @property
    def top_points(self):
        """Returns a NodeSet object of all top-layer points."""
        pset = np.argwhere(self._layertyp == SurfaceMesh.LAYER_TOP).T[0]
        return NodeSet("TopNodes", self.node_id[pset])

    @property
    def bottom_faces(self):
        """Returns a SideSet object of all bottom-layer faces."""
        eset = np.argwhere(self._layertyp_elem == SurfaceMesh.LAYER_BOTTOM).T[0]
        return SideSet("BottomFaces", self.cell_id[eset], self.face_id[eset])

    @property
    def side_faces(self):
        """Returns a SideSet object of all side-layer faces."""
        eset = np.argwhere(self._layertyp_elem == SurfaceMesh.LAYER_SIDES).T[0]
        return SideSet("SideFaces", self.cell_id[eset], self.face_id[eset])


def extract_surface_mesh(mesh):
    """
    Extracts the boundary of a mesh. For a solid (volume) mesh,
    it extracts the surface mesh. If it is a surface mesh, it
    extracts the edge mesh.

    Returns in Meshio format.
    """

    # REFERENCE: https://lagrit.lanl.gov/docs/EXTRACT_SURFMESH.html
    # ================================================================ #
    # Six new element based attributes, itetclr0, itetclr1, idelem0 and
    # idelem1, idface0, idface1 are added to the output mesh indicating
    # the material numbers (itetclr) on each side of the mesh faces,
    # i.e., the color of elements that existed on each side of a
    # face in the original mesh. The convention is that the normal
    # points into the larger material id (itetclr) material. itetclr0
    # indicates the color of the elements on smaller itetclr value side
    # (the inside) of the face normal (0 if the face is on an external
    # boundary) and itetclr1 indicates the color of the elements on
    # the outside of the normal. The attributes idelem0 and idelem1
    # record the element numbers of the input mesh that produced
    # the lower dimensional output mesh. The attributes idface0
    # and idface1 record the local face number of the input mesh objects.
    # A node attribute, idnode0, records the node number of the input
    # mesh object node.

    # In addition another element attribute called facecol is added to
    # the elements of the new mesh. The facecol attribute is a model
    # face number constructed from the itetclr0 and itetclr1 attributes.
    # The way the facecol  attribute is constructed does not guarantee
    # that the same facecol value will not be given to two disjoint
    # patches of mesh faces.

    with tempfile.TemporaryDirectory() as tmp_dir:
        mesh.save(os.path.join(tmp_dir, "volmesh.inp"))

        debug("Launching PyLaGriT")
        lg = PyLaGriT(verbose=_pylagrit_verbosity(), cwd=tmp_dir)

        cmds = [
            # (1) Read in the volume mesh
            "read/avs/volmesh.inp/mo_vol",
            "quality",
            # (2) Extract the external surface mesh
            "extract/surfmesh/1,0,0/mo_surf/mo_vol/external",
            # (3) Map the layertyp node attribute to elements:
            # (3.1) Create an element-based attribute
            "cmo/addatt/mo_surf/layertypelem/VINT/scalar/nelements",
            "cmo/setatt/mo_surf/layertypelem/1,0,0/0",
            # (3.2) Create pointsets for the top & bottom layers of points
            "pset/ptop/attribute/layertyp/1,0,0/-2/eq",
            "pset/pbot/attribute/layertyp/1,0,0/-1/eq",
            # (3.3) Capture the elements on the top & bottom layers
            "eltset/etop/exclusive/pset,get,ptop",
            "eltset/ebot/exclusive/pset,get,pbot",
            # (3.4) Set the element attribute to -2 for top and -1 for bottom
            "cmo/setatt/mo_surf/layertypelem/eltset,get,etop/-2",
            "cmo/setatt/mo_surf/layertypelem/eltset,get,ebot/-1",
            # (4) Write surface mesh to disk
            "dump/avs/surfmesh_lg.inp/mo_surf",
        ]

        for cmd in cmds:
            lg.sendline(cmd)

        del lg

        surf_mesh = meshio.read(
            os.path.join(tmp_dir, "surfmesh_lg.inp"), file_format="avsucd"
        )

    return surf_mesh
