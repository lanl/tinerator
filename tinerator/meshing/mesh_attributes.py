import numpy as np
from typing import Union
from ..logging import debug, warn


class MeshAttribute:

    NODE_TYPE_ALIAS = ("node", "point", "vertex", "vertices")
    CELL_TYPE_ALIAS = ("cell", "element", "voxel")
    SCALAR_TYPE_ALIAS = "scalar"

    INTEGER_TYPE = int
    FLOAT_TYPE = float

    INTEGER_TYPE_ALIAS = ("integer", "int", int, np.int, np.int64)
    FLOAT_TYPE_ALIAS = ("float", "double", "real", float, np.double)

    NODE_TYPE = "node"
    CELL_TYPE = "cell"
    SCALAR_TYPE = "scalar"

    def __init__(
        self,
        name: str = None,
        attribute_type: str = None,
        data: Union[int, float, np.array] = None,
        data_type: Union[type, str] = None,
        is_private: bool = False,
        num_mesh_nodes: int = None,
        num_mesh_cells: int = None,
    ):
        """
        Stores information on mesh attributes.

        Args
        ----
            name (str): The attribute name.
            attribute_type (str): "cell" or "node".
            data (Any): The data for the attribute.
            data_type (Union[type, str]): The type (int, float) of the data.
            is_private (bool): Marks whether it's an internal variable or public variable.
        """

        debug(
            f'Initializing attribute. Name: "{name}"; is_private: {is_private} '
            f"Attribute type: {attribute_type}; data: {type(data)}"
        )

        assert isinstance(name, str), "Attribute must have a name"
        assert isinstance(
            is_private, bool
        ), "Attribute must be marked as private or public"

        self.name = name
        self.attribute_type = attribute_type
        self._data = data
        self.data_type = data_type
        self.is_private = is_private
        self._num_parent_cells = num_mesh_cells
        self._num_parent_nodes = num_mesh_nodes

        self._set_attribute_type()
        self._set_data_type()

        debug(
            f"Attribute params: name = {name}; "
            f"attribute_type = {attribute_type} -> {self.attribute_type}; "
            f"data = {data} -> {np.unique(self.data)}; "
            f"data_type = {data_type} -> {self.data_type}; "
            f"is_private = {is_private}"
        )

    @property
    def data(self):
        """Returns properly formatted data vector."""

        def process_data(data, target_size, dtype):
            if isinstance(data, np.ndarray):
                return data.astype(self.data_type)
            elif isinstance(data, (tuple, list)):
                return np.array(data, dtype=self.data_type)
            elif isinstance(data, (int, np.int, np.int64, float, np.double)):
                return np.full((target_size,), data, dtype=dtype)
            else:
                warn("Could not parse data")
                return np.array(data).astype(dtype)

        attribute_type = self.attribute_type
        if attribute_type == MeshAttribute.SCALAR_TYPE:
            return self._data
        elif attribute_type == MeshAttribute.CELL_TYPE:
            return process_data(self._data, self._num_parent_cells, self.data_type)
        elif attribute_type == MeshAttribute.NODE_TYPE:
            return process_data(self._data, self._num_parent_nodes, self.data_type)
        else:
            raise ValueError("Could not set data vector")

    @data.setter
    def data(self, value):
        """Sets the data value."""
        if self.attribute_type in [MeshAttribute.CELL_TYPE, MeshAttribute.NODE_TYPE]:
            target_size = (
                self._num_parent_cells
                if self.attribute_type == MeshAttribute.CELL_TYPE
                else self._num_parent_nodes
            )
            if isinstance(value, (tuple, list, np.ndarray)):
                assert len(value) == target_size

        self._data = value

    def _set_data_type(self):
        data_type = self.data_type

        if self.attribute_type != MeshAttribute.SCALAR_TYPE:
            if data_type is None:
                try:
                    data_type = type(self._data[0])
                except TypeError:
                    data_type = type(self._data)

            if data_type in MeshAttribute.INTEGER_TYPE_ALIAS:
                data_type = MeshAttribute.INTEGER_TYPE
            elif data_type in MeshAttribute.FLOAT_TYPE_ALIAS:
                data_type = MeshAttribute.FLOAT_TYPE
            else:
                raise AttributeError(
                    f"Data type must be int or float, not: {data_type}"
                )

    def _set_attribute_type(self):
        """Attempts to set an attribute type from the data."""
        attribute_type = self.attribute_type
        data = self.data

        if attribute_type is None:
            try:
                data_len = len(data)
            except TypeError:
                raise TypeError(
                    f"Could not parse correct attribute type from data provided"
                )

            if data_len == self._num_parent_cells:
                attribute_type = MeshAttribute.CELL_TYPE
            elif data_len == self._num_parent_nodes:
                attribute_type = MeshAttribute.NODE_TYPE
            else:
                raise ValueError(
                    "Could not parse correct attribute type from data provided"
                )
        elif attribute_type in MeshAttribute.SCALAR_TYPE_ALIAS:
            attribute_type = MeshAttribute.SCALAR_TYPE
        elif attribute_type in MeshAttribute.CELL_TYPE_ALIAS:
            attribute_type = MeshAttribute.CELL_TYPE
        elif attribute_type in MeshAttribute.NODE_TYPE_ALIAS:
            attribute_type = MeshAttribute.NODE_TYPE
        else:
            raise ValueError(f"Could not parse attribute type: {attribute_type}")

        self.attribute_type = attribute_type

    @property
    def is_node_attribute(self):
        if self.attribute_type == MeshAttribute.NODE_TYPE:
            return True
        return False

    @property
    def is_cell_attribute(self):
        if self.attribute_type == MeshAttribute.CELL_TYPE:
            return True
        return False

    @property
    def is_scalar_attribute(self):
        if self.attribute_type == MeshAttribute.SCALAR_TYPE:
            return True
        return False

    def set_data(self, value):
        """
        Changes the data of the attribute to ``value``.
        """
        self.data = value
