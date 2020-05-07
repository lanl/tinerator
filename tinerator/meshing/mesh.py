import numpy as np
from copy import deepcopy
from enum import Enum, auto
from .io import write_avs
from ..visualize import view_3d as v3d


class ElementType(Enum):
    TRIANGLE = auto()
    QUAD = auto()
    PRISM = auto()
    HEX = auto()


# TODO: enable face and edge information. https://pymesh.readthedocs.io/en/latest/basic.html#mesh-data-structure


class Mesh:
    def __init__(
        self,
        name: str = "mesh",
        nodes: np.ndarray = None,
        elements: np.ndarray = None,
        etype: np.ndarray = None,
    ):
        self.name = name
        self.nodes = nodes
        self.elements = elements
        self.element_type = etype
        self.metadata = {}
        self.attributes = {}

    def __repr__(self):
        return f"Mesh<name: \"{self.name}\", nodes: {self.n_nodes}, elements<{self.element_type}>: {self.n_elements}>"

    def get_attribute(self, name: str):
        try:
            return self.attributes[name]["data"]
        except KeyError:
            raise KeyError("Attribute '%s' does not exist" % name)

    def add_attribute(
        self, name: str, vector: np.ndarray, attrb_type: str = None
    ):
        # TODO: change 'cell' to 'elem' for consistency or vice versa
        # TODO: auto-add attribute as cell or node based on array length
        if name in self.attributes:
            raise KeyError("Attribute %s already exists" % name)

        if not isinstance(vector, np.ndarray):
            vector = np.array(vector)

        # Take a guess at the attribute type
        if attrb_type is None:
            if vector.shape[0] == self.n_elements:
                attrb_type = "cell"
            elif vector.shape[0] == self.n_nodes:
                attrb_type = "node"

        attrb_type = attrb_type.lower().strip()

        if attrb_type not in ["cell", "node"]:
            raise ValueError("`attrb_type` must be either 'cell' or 'node'")

        sz = self.n_elements if attrb_type == "cell" else self.n_nodes
        vector = np.reshape(vector, (sz,))

        self.attributes[name] = {"type": attrb_type, "data": vector}

    def rm_attribute(self, name: str):
        try:
            del self.attributes[name]
        except KeyError:
            raise KeyError("Attribute '%s' does not exist" % name)

    @property
    def material_id(self):
        """Material ID of mesh"""
        return self.get_attribute("material_id")

    @property
    def x(self):
        """X vector of mesh nodes"""
        if self.nodes is not None:
            return self.nodes[:, 0]
        return None

    # TODO: check if this works. Does it work with slicing too?
    @x.setter
    def x(self, x):

        x = np.array(x)

        if self.nodes is not None:
            self.nodes[:, 0] = np.reshape(x, (self.n_nodes, 1))
        else:
            self.nodes = np.zeros((len(x), 3), dtype=float)
            self.nodes[:, 0] = x

    @property
    def y(self):
        """Y vector of mesh nodes"""
        if self.nodes is not None:
            return self.nodes[:, 1]
        return None

    @property
    def z(self):
        """Z vector of mesh nodes"""
        if self.nodes is not None:
            return self.nodes[:, 2]
        return None

    @property
    def n_nodes(self):
        """Number of nodes in mesh"""
        if self.nodes is not None:
            return self.nodes.shape[0]
        return 0

    @property
    def n_elements(self):
        """Number of elements in mesh"""
        if self.elements is not None:
            return self.elements.shape[0]
        return 0

    @property
    def centroid(self):
        """Mesh centroid"""
        if self.nodes is not None:
            return (np.mean(self.x), np.mean(self.y), np.mean(self.z))
        return None

    @property
    def extent(self):
        """
        Returns the extent of the mesh: 
            [ (x_min, x_max), (y_min, y_max), (z_min, z_max) ]
        """

        if self.nodes is None:
            return None

        ex = []
        for i in range(3):
            vector = self.nodes[:, i]
            ex.append((np.min(vector), np.max(vector)))

        return ex
    
    def view(self, attribute_name:str=None, scale:tuple=(1,1,1), **kwargs):
        '''
        Views the mesh object in an interactive VTK-rendered windowed environment.

        `**kwargs` are passed through to pyvista.UnstructuredGrid.plot method. Some keyword args include:

        * full_screen (bool, default: False)
        * window_size (tuple)
        * notebook (bool, default: False)
        * text (str, default: '')

        See `help(pyvista.UnstructuredGrid.plot)` and `help(pyvista.Plotter.add_mesh)` for more 
        information on possible keyword arguments.
        '''

        if self.element_type == ElementType.TRIANGLE:
            etype = 'tri'
        elif self.element_type == ElementType.PRISM:
            etype = 'prism'
        else:
            raise ValueError("Unknown `self.element_type`...is mesh object malformed?")
        
        try:
            matid = self.material_id
            cell_arrays = { 'materialID': matid }
        except KeyError:
            cell_arrays = None

        v3d.plot_3d(self, etype, cell_arrays=cell_arrays, scale=scale, **kwargs)
        
    def save(self, outfile: str):

        if self.element_type == ElementType.TRIANGLE:
            cell_type = "tri"
        elif self.element_type == ElementType.PRISM:
            cell_type = "prism"
        elif self.element_type is None:
            cell_type = None
        else:
            raise ValueError("Unknown cell type")

        try:
            mat_id = self.material_id
        except KeyError:
            mat_id = None

        node_attributes = {}
        for attr in self.attributes:
            if self.attributes[attr]["type"] == "node":
                node_attributes[attr] = {
                    "data": self.attributes[attr]["data"],
                    "type": "integer",
                }

        write_avs(
            outfile,
            self.nodes,
            self.elements,
            cname=cell_type,
            matid=mat_id,
            node_attributes=node_attributes,
        )

'''
class StackedMesh(Mesh):
    def __init__(self):
        self.number_of_layers = None
        self.
'''