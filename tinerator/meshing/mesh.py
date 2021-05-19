import meshio
import os
import shutil
import numpy as np
from pyproj import CRS
from pylagrit import PyLaGriT
from .dump_exodus import dump_exodus
from .mesh_metrics import triangle_quality
from .facesets_lg import write_facesets
from .readwrite import ElementType, read_avs, write_avs, read_mpas
from .surface_mesh import SurfaceMesh
from ..visualize import view_3d as v3d
from ..visualize import plot_triangulation
from ..logging import error, print_histogram_table


def _get_driver(filename: str):
    """Helper function for parsing mesh driver from filename."""
    ext = os.path.splitext(filename)[-1].replace(".", "").lower().strip()

    if ext in ["inp", "avs"]:
        return "avsucd"
    elif ext in ["nc", "mpas"]:
        return "mpas"
    elif ext in ["exo", "ex"]:
        return "exodus"
    elif ext in ["vtk", "vtu"]:
        return "vtk"
    else:
        return None


def load_mesh(
    filename: str,
    load_dual_mesh: bool = True,
    block_id: int = None,
    driver: str = None,
    name: str = None,
):
    """
    Loads a Mesh object from a mesh file on disk. Supports AVS-UCD, VTK, Exodus, and MPAS.
    """

    if driver is None:
        driver = _get_driver(filename)

    if name is None:
        name = os.path.basename(filename)

    mesh_object = Mesh()

    if driver == "mpas":
        mesh_object.nodes, mesh_object.elements = read_mpas(
            filename, load_dual_mesh=load_dual_mesh
        )

        if load_dual_mesh:
            mesh_object.element_type = ElementType.TRIANGLE
        else:
            mesh_object.element_type = ElementType.POLYGON

    elif driver == "avsucd":
        nodes, elements, element_type, material_id, node_atts, cell_atts = read_avs(
            filename
        )
        mesh_object.nodes = nodes
        mesh_object.elements = elements
        mesh_object.element_type = element_type

        if material_id is not None:
            mesh_object.material_id = material_id

        if node_atts is not None:
            for att in node_atts.keys():
                mesh_object.add_attribute(att, node_atts[att], attrb_type="node")

        if cell_atts is not None:
            for att in cell_atts.keys():
                mesh_object.add_attribute(att, cell_atts[att], attrb_type="cell")
    else:
        mesh = meshio.read(filename, file_format=driver)
        mesh_object.nodes = mesh.points

        if block_id is None:
            block_id = 0

        mesh_object.elements = mesh.cells[block_id].data + 1

        if mesh.cells[block_id].type == "triangle":
            mesh_object.element_type = ElementType.TRIANGLE
        elif mesh.cells[block_id].type == "wedge":
            mesh_object.element_type = ElementType.PRISM
        else:
            raise ValueError("Mesh type is currently not supported.")

    return mesh_object


# TODO: enable face and edge information. https://pymesh.readthedocs.io/en/latest/basic.html#mesh-data-structure


class Mesh:
    def __init__(
        self,
        name: str = "mesh",
        nodes: np.ndarray = None,
        elements: np.ndarray = None,
        etype: ElementType = None,
        crs: CRS = None,
    ):
        self.name = name
        self.nodes = nodes
        self.elements = elements
        self.element_type = etype
        self.crs = crs
        self.metadata = {}
        self.attributes = {}

    def __repr__(self):
        return f'Mesh<name: "{self.name}", nodes: {self.n_nodes}, elements({self.element_type}): {self.n_elements}>'

    def get_attribute(self, name: str):
        try:
            return self.attributes[name]["data"]
        except KeyError:
            raise KeyError("Attribute '%s' does not exist" % name)

    def set_attribute(self, name: str, vector: np.ndarray, at_layer: int = None):
        if at_layer is not None:
            if self.element_type != ElementType.PRISM:
                raise ValueError(f"`at_layer` not supported for {self.element_type}.")

            if at_layer < 0:
                at_layer = self._num_layers + at_layer + 1

            if self.attributes[name]["type"] == "cell":
                ln = self._elems_per_layer
                ln_full = self.n_elements
            elif self.attributes[name]["type"] == "node":
                ln = self._nodes_per_layer
                ln_full = self.n_nodes
            else:
                raise ValueError("Unknown attribute type.")

            if isinstance(vector, np.ndarray):
                vector = vector.flatten()

                if vector.shape[0] != ln:
                    raise ValueError(
                        f"Requires a vector of length {ln}. Was given length {vector.shape[0]}."
                    )
            elif isinstance(vector, (int, float)):
                vector = np.full((ln,), vector)

            end = ln_full - at_layer * ln
            start = ln_full - (at_layer + 1) * ln

            assert end - start == ln
        else:
            start = 0
            end = ln

        self.attributes[name]["data"][start:end] = vector

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

        vector = np.full((sz,), fill_value)

        self.add_attribute(name, vector, attrb_type=attrb_type)

    def add_attribute(self, name: str, vector: np.ndarray, attrb_type: str = None):
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

    def reset_attributes(self):
        self.attributes = {}

    def get_cell_centroids(self):
        """Compute the centroids of every cell"""

        # TODO: optimize function
        centroids = np.zeros((self.n_elements, 3), dtype=float)

        for (i, elem) in enumerate(self.elements):
            centroids[i] = np.mean(self.nodes[elem - 1], axis=0)

        return centroids

    def surface_mesh(self):
        """
        Extracts the surface mesh: a mesh where all interior voxels
        and faces are removed. For triangular meshes, this will
        return a line mesh. For a prism (extruded) mesh, this will
        return a mesh containing triangles and quads.

        Returns in Meshio format.
        """
        return SurfaceMesh(self)

    def mesh_quality(self, plot=False, n_bins: int = None):
        """
        Displays the quality of the mesh.
        If ``plot = True``, will display a plot showing the triangle
        quality and a histogram.
        If ``plot = False``, will print out an ASCII table displaying
        binned quality values.

        Args
        ----
            plot (:obj:`bool`, optional): Whether to plot results or display as ASCII.
            n_bins (:obj:`int`, optional): Overrides the default number of bins for histogram binning.

        Examples
        --------
            >>> triangle_mesh.mesh_quality(plot=False)
            >>> triangle_mesh.mesh_quality(plot=True, n_bins=11)
        """
        if self.element_type != ElementType.TRIANGLE:
            raise NotImplementedError(
                "Currently only implemented for triangular meshes."
            )

        quality = triangle_quality(self)

        if plot:
            if n_bins is None:
                n_bins = 30
            plot_triangulation(
                self.nodes,
                self.elements - 1,
                face_attribute=quality,
                title="Triangle Quality",
                histogram_bins=n_bins,
                histogram_range=(0.0, 1.0),
            )
        else:
            if n_bins is None:
                n_bins = 11
            q_hist, q_bins = np.histogram(
                quality, bins=np.linspace(0.0, 1.0, num=n_bins, endpoint=True)
            )
            print_histogram_table(q_hist, q_bins, title="Triangle Quality")

    @property
    def material_id(self):
        """Material ID of mesh"""
        try:
            return self.get_attribute("material_id")
        except KeyError:
            v = np.ones((self.n_elements,), dtype=int)
            self.material_id = v
            return self.get_attribute("material_id")

    @material_id.setter
    def material_id(self, v):
        self.add_attribute("material_id", np.array(v, dtype=int), attrb_type="cell")

    @property
    def node_attributes(self):
        """Returns available node attributes"""
        atts = self.attributes
        return [x for x in atts if atts[x]["type"] == "node"]

    @property
    def element_attributes(self):
        """Returns available element (cell) attributes"""
        atts = self.attributes
        return [x for x in atts if atts[x]["type"] == "cell"]

    def map_raster_to_attribute(
        self,
        raster,
        attribute_name: str = "material_id",
        attribute_type: str = "cell",
    ):
        """
        Maps a Raster object to a mesh attribute.
        """

        # This *only* works for surfaces
        if attribute_type == "cell":
            points = self.get_cell_centroids()
        elif attribute_type == "node":
            points = self.nodes

        vector = raster.values_at(points)
        self.add_attribute(attribute_name, vector, attrb_type=attribute_type)

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

    @property
    def edges(self):
        """
        Returns all unique edges in the mesh.
        """

        # TODO: this is probably not the most efficient algorithm

        v_edges = [np.vstack([self.elements[:, -1], self.elements[:, 0]]).T]

        for i in range(self.elements.shape[-1] - 1):
            v_edges.append(self.elements[:, i : i + 2])

        return np.unique(np.sort(np.vstack(v_edges), axis=1), axis=0)

    def view(
        self,
        active_scalar: str = None,
        scale: tuple = (1, 1, 1),
        savefig: str = None,
        show_bounds: bool = True,
        show_edges: bool = True,
        window_size: tuple = None,
        **kwargs,
    ):
        """
        Views the mesh object in an interactive VTK-rendered windowed environment.
        In a Jupyter notebook, it will render the 3D mesh in a live cell.

        Additional keyword arguments can be found in the PyVista documentation.

        https://docs.pyvista.org/plotting/plotting.html#pyvista.plot

        Args
        ----
            active_scalar (:obj:`str`, optional): The mesh attribute to visualize. Defaults to material ID.
            scale (:obj:`tuple`, optional): Relative scale for the mesh in (X, Y, Z). Defaults to (1, 1, 1).
            savefig (:obj:`str`, optional): Filepath to save a screenshot of the mesh.
            show_bounds (:obj:`bool`, optional): If True, shows the bounding box of the mesh.
            show_edges (:obj:`bool`, optional): If True, shows the mesh edges.
            window_size (:obj:`bool`, optional): Adjusts the viewing window size or the Jupyter notebook cell size.
        """

        if self.element_type == ElementType.TRIANGLE:
            etype = "tri"
        elif self.element_type == ElementType.PRISM:
            etype = "prism"
        elif self.element_type == ElementType.POLYGON:
            etype = "polygon"
        else:
            raise ValueError("Unknown `self.element_type`...is mesh object malformed?")

        cell_arrays = None
        node_arrays = None

        try:
            if active_scalar:
                attrb = self.get_attribute(active_scalar)
                attribute_name = active_scalar
            else:
                attrb = self.material_id
                attribute_name = "material-ID"

            if len(attrb) == self.n_nodes:
                node_arrays = {attribute_name: attrb}
            elif len(attrb) == self.n_elements:
                cell_arrays = {attribute_name: attrb}
            else:
                raise ValueError("Malformed attribute vector")
        except KeyError:
            error(f'Could not find attribute "{active_scalar}"')

        v3d.plot_3d(
            self,
            etype,
            active_scalar=attribute_name,
            cell_arrays=cell_arrays,
            node_arrays=node_arrays,
            scale=scale,
            text=f"Nodes: {self.n_nodes}\nCells: {self.n_elements}",
            screenshot=savefig,
            show_edges=show_edges,
            window_size=window_size,
            show_bounds=show_bounds,
            **kwargs,
        )

    def save(
        self,
        outfile: str,
        face_sets: list = None,
        node_sets: list = None,
        element_sets: list = None,
    ):
        """
        Writes a mesh object to disk. Supports file formats like VTK, AVS-UCD, and Exodus.

        For Exodus meshes, face sets, node sets, and elements sets can be written out with the mesh.

        Args:
            outfile (str): The path to save the mesh. The file format of the mesh is automatically inferred by the extension.
            face_sets (:obj:`list`, optional): A list of face sets.
            node_sets (:obj:`list`, optional): A list of node sets.
            element_sets (:obj:`list`, optional): A list of element sets.

        Examples:
            >>> tri = tin.meshing.triangulate(my_dem, min_edge_length=0.005)
            >>> tri.save("my_triangulation.vtk")
            >>> tri.save("my_triangulation.inp")
            >>> tri.save("my_triangulation.exo", face_sets = face_sets)
        """

        driver = _get_driver(outfile)

        if driver == "exodus":
            dump_exodus(
                outfile,
                self.nodes,
                self.elements,
                side_sets=face_sets,
                node_sets=node_sets,
                element_sets=element_sets,
            )
        elif driver == "avsucd":
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

            node_attributes = {x: self.get_attribute(x) for x in self.node_attributes}
            elem_attributes = {
                x: self.get_attribute(x)
                for x in self.element_attributes
                if x != "material_id"
            }

            write_avs(
                outfile,
                self.nodes,
                self.elements,
                cname=cell_type,
                matid=mat_id,
                node_attributes=node_attributes,
                cell_attributes=elem_attributes,
            )
        else:
            self.as_meshio().write(outfile)

    def as_meshio(self, material_id_as_cell_blocks: bool = False) -> meshio.Mesh:
        """Converts a Mesh object into a meshio.Mesh object."""

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
                element = self.elements[i, :]
                cells.append(("polygon", element[element > 0] - 1))

        else:
            cells = [(cell_type, self.elements - 1)]

        # TODO: this needs to support every attribute!
        # TODO: this needs to take into account `material_id_as_cell_blocks`
        cell_data = {"materialID": [self.material_id]}

        mesh = meshio.Mesh(
            points=self.nodes,
            cells=cells,
            cell_data=cell_data,
            point_data=None,
        )

        return mesh


class StackedMesh(Mesh):
    def __init__(self, name: str = "stacked_mesh", etype: ElementType = None):
        super().__init__(name=name, etype=etype)
        self._cell_layer_ids = None
        self._num_layers = None
        self._nodes_per_layer = None
        self._elems_per_layer = None

    def get_cells_at_sublayer(self, sublayer: int) -> np.ndarray:
        raise NotImplementedError("Layering in progress")
        # return self._cell_layer_ids == layer
