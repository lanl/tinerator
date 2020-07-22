import meshio
import numpy as np
from copy import deepcopy
from enum import Enum, auto
from .readwrite import write_avs, read_mpas
from ..visualize import view_3d as v3d

def load(filename:str, load_dual_mesh:bool = True):
    nodes, cells = read_mpas(filename, load_dual_mesh=load_dual_mesh)

    m = Mesh()
    m.nodes = nodes
    m.elements = cells

    if load_dual_mesh:
        m.element_type = ElementType.TRIANGLE
    else:
        m.element_type = ElementType.POLYGON

    return m

class ElementType(Enum):
    TRIANGLE = auto()
    QUAD = auto()
    PRISM = auto()
    HEX = auto()
    POLYGON = auto()


# TODO: enable face and edge information. https://pymesh.readthedocs.io/en/latest/basic.html#mesh-data-structure


class Mesh:
    def __init__(
        self,
        name: str = "mesh",
        nodes: np.ndarray = None,
        elements: np.ndarray = None,
        etype: ElementType = None,
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

    def add_empty_attribute(self, name: str, attrb_type: str, fill_value: float = 0.0):
        """
        Creates an empty cell or node attribute with an optional given fill value.
        """

        if attrb_type.lower().strip() == "cell":
            sz = self.n_elements
        elif attrb_type.lower().strip() == "node":
            sz = self.n_nodes
        else:
            raise ValueError(f"Unknown attrb_type: {attrb_type}")
        
        vector = np.full((sz,),fill_value)

        self.add_attribute(name, vector, attrb_type=attrb_type)

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

    def get_cell_centroids(self):
        """Compute the centroids of every cell"""

        # TODO: optimize function
        centroids = np.zeros((self.n_elements,3), dtype=float)

        for (i,elem) in enumerate(self.elements):
            centroids[i] = np.mean(self.nodes[elem - 1], axis=0)

        return centroids

    @property
    def material_id(self):
        """Material ID of mesh"""
        try:
            return self.get_attribute("material_id")
        except KeyError:
            v = np.ones((self.n_elements,),dtype=int)
            self.material_id = v
            return self.get_attribute("material_id")
    
    @material_id.setter
    def material_id(self, v):
        self.add_attribute("material_id", v, attrb_type="cell")

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
        elif self.element_type == ElementType.POLYGON:
            etype = 'polygon'
        else:
            raise ValueError("Unknown `self.element_type`...is mesh object malformed?")
        
        cell_arrays = None
        node_arrays = None

        try:
            if attribute_name:
                attrb = self.get_attribute(attribute_name)
            else:
                attrb = self.material_id
                attribute_name = "materialID"

            if len(attrb) == self.n_nodes:
                node_arrays = { attribute_name: attrb }
            elif len(attrb) == self.n_elements:
                cell_arrays = { attribute_name: attrb }
            else:
                raise ValueError("Malformed attribute vector")
        except KeyError:
            pass

        v3d.plot_3d(self, etype, cell_arrays=cell_arrays, node_arrays=node_arrays, scale=scale, **kwargs)
        
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
    
    def as_meshio(self,material_id_as_cell_blocks:bool = False) -> meshio.Mesh:
        '''Converts a Mesh object into a meshio.Mesh object.'''

        if self.element_type == ElementType.TRIANGLE:
            cell_type = "triangle"
        elif self.element_type == ElementType.PRISM:
            cell_type = "wedge"
        elif self.element_type == ElementType.POLYGON:
            cell_type = "polygon"
        else:
            raise ValueError("Unknown cell type")
        
        # Each unique value of material ID becomes a seperate
        # cell block - useful for Exodus exports
        if material_id_as_cell_blocks:
            elements = self.elements - 1
            mat_id = self.material_id
            cells = []

            for value in np.unique(mat_id):
                cells.append((cell_type, elements[mat_id == value]))
        elif cell_type == "polygon":
            cells = []

            for i in range(self.n_elements):
                element = self.elements[i,:]
                cells.append(("polygon", element[element > 0] - 1))

        else:
            cells = [
                (cell_type, self.elements - 1)
            ]

        # TODO: this needs to support every attribute!
        # TODO: this needs to take into account `material_id_as_cell_blocks`
        cell_data = { 
            "materialID": self.material_id
        }

        mesh = meshio.Mesh(
            points=self.nodes, 
            cells=cells, 
            cell_data=cell_data,
            point_data=None,
        )

        return mesh


    def save_exo(self, outfile:str):
        '''Uses the Meshio library as a driver for file output.'''
        meshio.write(outfile, self.as_meshio(material_id_as_cell_blocks=True))        

class StackedMesh(Mesh):
    def __init__(self, name:str="stacked_mesh", etype: ElementType = None):
        super().__init__(name=name,etype=etype)

        self._cell_layer_ids = None
        self._num_layers = None
        self._nodes_per_layer = None
        self._elems_per_layer = None
    
    def get_cells_at_sublayer(sublayer: int) -> np.ndarray:
        return self._cell_layer_ids == layer
    
