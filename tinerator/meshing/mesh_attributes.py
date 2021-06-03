import numpy as np
from typing import Union
from ..logging import debug, warn

class MeshAttribute:

    NODE_TYPE_ALIAS = ("node", "point", "vertex", "vertices")
    CELL_TYPE_ALIAS = ("cell", "element", "voxel")
    SCALAR_TYPE_ALIAS = ("scalar")

    INTEGER_TYPE = int
    FLOAT_TYPE = float

    INTEGER_TYPE_ALIAS = ("integer", "int", int, np.int)
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
            f"Initializing attribute. Name: \"{name}\"; is_private: {is_private} "
            f"Attribute type: {attribute_type}; data: {type(data)}"
        )

        assert isinstance(name, str), "Attribute must have a name"
        assert isinstance(is_private, bool), "Attribute must be marked as private or public"

        if attribute_type is None:
            attribute_type = ''

        if attribute_type in MeshAttribute.SCALAR_TYPE_ALIAS:
            attribute_type = MeshAttribute.SCALAR_TYPE
        else:
            if attribute_type in [*MeshAttribute.CELL_TYPE_ALIAS, *MeshAttribute.NODE_TYPE_ALIAS]:
                if attribute_type in MeshAttribute.CELL_TYPE_ALIAS:
                    target_length = num_mesh_cells
                    attribute_type = MeshAttribute.CELL_TYPE
                elif attribute_type in MeshAttribute.NODE_TYPE_ALIAS:
                    target_length = num_mesh_nodes
                    attribute_type = MeshAttribute.NODE_TYPE
                
                if isinstance(data, np.ndarray):
                    data = data # do nothing
                elif isinstance(data, (int, float)):
                    data = np.full((target_length,), data)
                elif isinstance(data, (tuple, list)):
                    data = np.array(data)
                else:
                    warn(f"Not sure how to parse data...")
                    data = np.array(data)
            else:
                try:
                    target_length = len(data)
                except TypeError as e:
                    raise AttributeError(e)
                
                if target_length == num_mesh_cells:
                    attribute_type = MeshAttribute.CELL_TYPE
                elif target_length == num_mesh_nodes:
                    attribute_type = MeshAttribute.NODE_TYPE
                else:
                    raise AttributeError(
                        "Length of data does not match number of nodes or cells,"
                        "and data is not a scalar type"
                    )
            
            if attribute_type == MeshAttribute.CELL_TYPE:
                assert len(data) == num_mesh_cells
            elif attribute_type == MeshAttribute.NODE_TYPE:
                assert len(data) == num_mesh_nodes
            else:
                raise AttributeError(f"Incorrect attribute type: {attribute_type}")

        # ===================================== #

        # Validate the data type: int or float
        if attribute_type != MeshAttribute.SCALAR_TYPE:
            if data_type is None:
                try:
                    data_type = type(data[0])
                except TypeError:
                    data_type = type(data)

            if data_type in MeshAttribute.INTEGER_TYPE_ALIAS:
                data_type = MeshAttribute.INTEGER_TYPE
            elif data_type in MeshAttribute.FLOAT_TYPE_ALIAS:
                data_type = MeshAttribute.FLOAT_TYPE
            else:
                raise AttributeError(f"Data type must be int or float, not: {data_type}")
        
        debug(
            f"Attribute params: name = {name}; "
            f"attribute_type = {attribute_type}; "
            f"data = {data}; "
            f"data_type = {data_type}; "
            f"is_private = {is_private}"
        )

        self.name = name
        self.attribute_type = attribute_type
        self.data = data
        self.data_type = data_type
        self.is_private = is_private
    
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
        if not isinstance(value, np.ndarray):
            raise ValueError(f"Could not support type {type(value)}")
        self.data = value
