from tempfile import TemporaryDirectory
import meshio
import os
import numpy as np
from pyproj import CRS
from typing import Union, List
from .write_exodusii_mesh import dump_exodus
from .mesh_metrics import triangle_quality
from .mesh_attributes import MeshAttribute
from .readwrite import read_avs, write_avs, read_mpas
from .meshing_types import ElementType
from .surface_mesh import SurfaceMesh
from ..visualize import view_3d as v3d
from ..visualize import plot_triangulation, plot_sets
from ..logging import error, print_histogram_table
from .sets import SideSet, ElementSet, PointSet


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
                mesh_object.add_attribute(att, node_atts[att], type="node")

        if cell_atts is not None:
            for att in cell_atts.keys():
                mesh_object.add_attribute(att, cell_atts[att], type="cell")
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


class Mesh:
    def __init__(
        self,
        nodes: np.ndarray = None,
        elements: np.ndarray = None,
        etype: ElementType = None,
        crs: CRS = None,
    ):
        if nodes is not None:
            nodes = np.array(nodes)

        if elements is not None:
            elements = np.array(elements)

        self.nodes = nodes
        self.elements = elements
        self.element_type = etype
        self.crs = crs
        self._attributes = []

    def __repr__(self):
        return f"Mesh<nodes: {self.n_nodes}, elements({self.element_type}): {self.n_elements}>"

    def add_attribute(
        self,
        name: str,
        value: Union[np.array, list, int, float],
        type: str = None,
        data_type: Union[type, str] = None,
        overwrite: bool = False,
        force: bool = False,
        **kwargs,
    ):
        """
        Adds a new attribute to the mesh. The ``name`` argument gives the attribute a name, and must
        be a string.

        The ``value`` argument can be a float (``3.14159``), an integer (``1``), or a vector (list/tuple/Numpy)
        of floats or integers. If ``value`` is a vector, then the length of the vector must match exactly
        to the number of nodes in the mesh (``self.num_nodes``) or to the number of cells in the mesh (``self.num_cells``).

        The optional ``type`` argument explicitly defines the type of attribute it should be:

        - "cell": The attribute is a cell (a.k.a. element or voxel) attribute.
        - "node": The attribute is a node (a.k.a. point or vertex) attribute.
        - "scalar": The attribute is a single value, which might be used for book-keeping (i.e., the filename of a mesh loaded from disk).

        The optional ``data_type`` argument specifies what the attribute type should be. It can be a literal
        Python type (like `int` or `float`), or a string representation of that type (like ``"float"`` or ``"integer"``).
        If ``type`` is ``"scalar"``, then ``data_type`` is ignored: this only matters for cell and node attributes.

        Args
        ----
            name (str): The name of the attribute to create.
            value (Union[np.array, list, int, float]): The value(s) to fill the attribute with.
            type (:obj:`str`, optional): The type of attribute to create. Choose ``"cell"`` or ``"node"``.
            data_type (:obj:`Union[type, str]`, optional): The data type of the attribute - ``float`` or ``int``.
            overwrite (:obj:`bool`, optional): If True, it will overwrite an existing attribute. If False, it will fail if the attribute exists.

        Throws
        ------
            AttributeError: if the attribute already exists.
        """

        if name in [*self.attributes, *self.private_attributes]:
            if overwrite:
                self.delete_attribute(name)
            else:
                raise AttributeError(f'Attribute "{name}" already exists.')

        is_private = kwargs["private"] if "private" in kwargs else False

        att = MeshAttribute(
            name=name,
            data=value,
            attribute_type=type,
            data_type=data_type,
            num_mesh_nodes=self.n_nodes,
            num_mesh_cells=self.n_elements,
            is_private=is_private,
        )
        self._attributes.append(att)

    def get_attribute(self, name: str):
        """
        Returns the data associated with the mesh attribute with name ``name``.

        Args
        ----
            name (str): The name of the attribute.

        Returns
        -------
            Any: the value of the attribute.

        Throws
        ------
            AttributeError: if the requested attribute doesn't exist.
        """

        for attribute in self._attributes:
            if attribute.name == name:
                return attribute.data

        raise AttributeError(f'Attribute "{name}" does not exist.')

    def set_attribute(self, name: str, value, at_layer: tuple = None):
        """
        Sets an existing attribute.
        """
        for attribute in self._attributes:
            if attribute.name == name:
                attribute.data = value
                return

        raise AttributeError('Attribute "{name}" does not exist')

    def delete_attribute(self, name: str, force: bool = False):
        """
        Deletes a mesh attribute.

        Args
        ----
            name (str): The name of the attribute to delete.
            force (:obj:`force`, optional): Required when attempting to delete internal attributes.

        Throws
        ------
            AttributeError: when the requested attribute cannot be found.
        """
        for (i, attribute) in enumerate(self._attributes):
            if attribute.name == name:
                if (force) or (not attribute.is_private):
                    self._attributes.pop(i)
                    return

        raise AttributeError(f'Attribute "{name}" does not exist')

    def remap_attribute(
        self,
        attribute_name: str,
        target_type: str,
        target_attribute_name: str = None,
        categorical_data: bool = False,
        target_data_type=None,
    ):
        """

        Args
        ----
            attribute_name (str): The attribute to remap.
            target_type (str): The type to map to. One of "cell", "node", or "scalar".
            target_attribute_name (:obj:`str`, optional): If not None, creates a new attribute with this name. If the same value as ``attribute_name``, then overwrites the existing attribute.
            target_data_type (:obj:`Union[type, str]`, optional): Changes the data type of the attribute to the one provided. One of "float", "int", or ``None``.
            categorical_data (:obj:`bool`, optional): For ``target_type = "cell"`` only. If True, then uses histogram binning to assign cell data values. If False, uses the average node value.

        Returns
        -------
            data: the remapped attribute.
        """
        cells = self.elements - 1

        if target_type != "cell":
            raise NotImplementedError()

        node_att = self.get_attribute(attribute_name)[cells]

        if categorical_data:
            median = np.median(node_att, axis=1)
            idx = (node_att - median[:, None]).argmin(
                axis=1
            )  # TODO: should allow argmax too
            new_data = node_att[np.arange(len(node_att)), idx]
        else:
            new_data = np.mean(node_att, axis=1)

        return new_data

    @property
    def private_attributes(self):
        """
        Returns the names of all private attributes. Private attributes are
        used for internal functionality.
        """
        return [att.name for att in self._attributes if att.is_private]

    @property
    def attributes(self):
        """
        Returns the names of all non-private attributes.
        """
        return [att.name for att in self._attributes if not att.is_private]

    @property
    def node_attributes(self):
        """
        Returns the names of all non-private attributes with a node type.
        """
        return [
            att.name
            for att in self._attributes
            if att.is_node_attribute and not att.is_private
        ]

    @property
    def cell_attributes(self):
        """
        Returns the names of all non-private attributes with a cell type.
        """
        return [
            att.name
            for att in self._attributes
            if att.is_cell_attribute and not att.is_private
        ]

    @property
    def element_attributes(self):
        """Alias for self.cell_attributes."""
        return self.cell_attributes

    @property
    def material_id(self):
        """
        Returns the material ID of each cell. The material ID
        is an integer number representing a unique material type.
        """
        try:
            return self.get_attribute("material_id")
        except AttributeError:
            self.material_id = 1
            return self.get_attribute("material_id")

    @material_id.setter
    def material_id(self, value):
        """
        Sets material ID to ``value``. For more complex functionality, use
        ``Mesh.set_attribute("material_id", *args, **kwargs)``.
        """
        try:
            self.set_attribute("material_id", value)
        except AttributeError:
            self.add_attribute(
                "material_id", value, type="cell", data_type=int, is_private=True
            )

    def add_attribute_from_raster(
        self,
        attribute_name: str,
        raster,
        attribute_type: str = "cell",
        data_type: Union[type, str] = None,
        **kwargs,
    ):
        """
        Creates a new attribute from raster data.
        If ``attribute_type = "cell"``, then every mesh cell will be filled with the values
        of the raster at the cell centroids of the mesh.
        If ``attribute_type = "node"``, then every mesh node will be filled with the values
        of the raster at the mesh nodes.
        The code does not currently support ``attribute_type = "scalar"``.

        Args
        ----
            attribute_name (str): The name of the attribute to create.
            raster (tinerator.gis.Raster): The Raster object to source attribute data from.
            attribute_type (:obj:`str`, optional): The type of attribute to create. "cell", "node", or "scalar.
            data_type (:obj:`Union[type, str]`, optional): Specifies the data type for the attribute. "int" or "float".
            **kwargs: Other keyword arguments for :obj:`Mesh.add_attribute`.
        """

        if attribute_type in MeshAttribute.CELL_TYPE_ALIAS:
            points = self.get_cell_centroids()
        elif attribute_type in MeshAttribute.NODE_TYPE_ALIAS:
            points = self.nodes
        elif attribute_type in MeshAttribute.SCALAR_TYPE_ALIAS:
            raise NotImplementedError("Scalars are not yet implemented.")

        data = raster.values_at(points)
        self.add_attribute(
            attribute_name, data, type=attribute_type, data_type=data_type
        )

    @property
    def metadata(self):
        """
        Returns the metadata of the mesh (a.k.a, mesh attributes with a scalar type) in
        a dictionary format.
        """
        return [
            {att.name: att.data} for att in self._attributes if att.is_scalar_attribute
        ]

    def get_cell_centroids(self):
        """Returns the centroids of every cell"""

        return np.mean(self.nodes[self.elements - 1], axis=1)

    def to_vtk_mesh(self, material_id_alias: str = None):
        """
        Returns the mesh in VTK/PyVista format.

        Args
        ----
            material_id_alias (:obj:`str`, optional): Renames the default PyVista name from "Material Id" to this.
        """
        import pyvista as pv

        with TemporaryDirectory() as tmp_dir:
            self.save(os.path.join(tmp_dir, "mesh.inp"))
            mesh = pv.read(os.path.join(tmp_dir, "mesh.inp"))

        if material_id_alias is not None:
            mesh[material_id_alias] = mesh["Material Id"]

        return mesh

    def surface_mesh(self):
        """
        Extracts the surface mesh: a mesh where all interior voxels
        and faces are removed. For triangular meshes, this will
        return a line mesh. For a prism (extruded) mesh, this will
        return a mesh containing triangles and quads.

        Returns in VTK/PyVista format.

        Returns
        -------
            surface_mesh: The surface (exterior) mesh.
        """

        surf_mesh = SurfaceMesh(self.to_vtk_mesh(material_id_alias="material_id"))

        if "layertyp" in self.attributes:
            surf_mesh.point_data_to_cell_data("layertyp", "layertyp_cell")

        return surf_mesh

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
        """Returns the centroid of the mesh"""
        return np.mean(self.get_cell_centroids(), axis=0)

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
        sets: Union[SideSet, ElementSet, PointSet] = None,
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

        Point Sets and Side Sets can also be rendered by passing a list of
        them to the ``sets`` keyword argument.

        Additional keyword arguments can be found in the PyVista documentation.

        https://docs.pyvista.org/plotting/plotting.html#pyvista.plot

        Args
        ----
            active_scalar (:obj:`str`, optional): The mesh attribute to visualize. Defaults to "material_id".
            sets (:obj:`Union[SideSet, PointSet]`, optional): The sets to render with the mesh.
            scale (:obj:`tuple`, optional): Relative scale for the mesh in (X, Y, Z). Defaults to (1, 1, 1).
            savefig (:obj:`str`, optional): Filepath to save a screenshot of the mesh.
            show_bounds (:obj:`bool`, optional): If True, shows the bounding box of the mesh.
            show_edges (:obj:`bool`, optional): If True, shows the mesh edges.
            window_size (:obj:`bool`, optional): Adjusts the viewing window size or the Jupyter notebook cell size.
        """

        if sets is not None:
            if active_scalar is None:
                active_scalar = "material_id"
                assert len(self.material_id) > 0

            plot_sets(
                self.to_vtk_mesh(material_id_alias="material_id"),
                sets,
                active_scalar=active_scalar,
                show_edges=show_edges,
                savefig=savefig,
                window_size=window_size,
                scale=scale,
                **kwargs,
            )
            return

        # TODO: use `self.to_vtk_mesh()` instead of what's below
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
            if active_scalar is None:
                active_scalar = "material_id"
                attrb = self.material_id
                attribute_name = active_scalar
            else:
                attrb = self.get_attribute(active_scalar)
                attribute_name = active_scalar

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

    def save_exodusii(
        self,
        outfile: str,
        sets: List[Union[SideSet, PointSet, ElementSet]] = None,
    ):
        """
        Writes the mesh to ExodusII format.
        Allows for more configuration than the standard
        :obj:`self.save()` method.

        Args:
            outfile (str): The path to save the mesh.
            sets (:obj:`List[Union[SideSet, PointSet]]`, optional): A list of side sets (face sets) and/or point sets.
        """

        if sets is not None:
            side_sets = [s.exodusii_format for s in sets if isinstance(s, SideSet)]
            node_sets = [s.exodusii_format for s in sets if isinstance(s, PointSet)]
            element_sets = None
        else:
            side_sets = None
            node_sets = None
            element_sets = None

        dump_exodus(
            outfile,
            self.nodes,
            self.elements,
            cell_block_ids=self.material_id,
            side_sets=side_sets,
            node_sets=node_sets,
            element_sets=element_sets,
        )

    def save(
        self, outfile: str, sets: List[Union[SideSet, PointSet, ElementSet]] = None
    ):
        """
        Writes a mesh object to disk. Supports file formats like VTK, AVS-UCD, and Exodus.

        For Exodus meshes, face sets, node sets, and elements sets can be written out with the mesh.

        Args:
            outfile (str): The path to save the mesh. The file format of the mesh is automatically inferred by the extension.
            sets (:obj:`List[Union[SideSet, PointSet, ElementSet]]`, optional): A list of side/point/element sets (works with ExodusII output only).

        Examples:
            >>> tri = tin.meshing.triangulate(my_dem, min_edge_length=0.005)
            >>> tri.save("my_triangulation.vtk")
            >>> tri.save("my_triangulation.inp")
            >>> tri.save("my_triangulation.exo", sets = [tri.top_faces])
        """

        driver = _get_driver(outfile)

        if driver == "exodus":
            self.save_exodusii(outfile, sets=sets)
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
    def __init__(self, etype: ElementType = None):
        super().__init__(etype=etype)
        self._cell_layer_ids = None
        self._num_layers = None
        self._nodes_per_layer = None
        self._elems_per_layer = None

    def get_cells_at_sublayer(self, sublayer: int) -> np.ndarray:
        raise NotImplementedError("Layering in progress")
        # return self._cell_layer_ids == layer
