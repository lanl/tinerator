import numpy as np
from typing import Union

class MeshAttribute:

    NODE_TYPE_ALIAS = ("node", "point", "vertex", "vertices")
    CELL_TYPE_ALIAS = ("cell", "element", "voxel")

    INTEGER_TYPE = int
    FLOAT_TYPE = float

    INTEGER_TYPE_ALIAS = ("integer", "int", int, np.int)
    FLOAT_TYPE_ALIAS = ("float", "double", "real", float, np.double)

    NODE_TYPE = "node"
    CELL_TYPE = "cell"

    def __init__(
            self, 
            name: str = None, 
            attribute_type: str = None, 
            data: Union[int, float, np.array] = None,
            data_type: Union[type, str] = None,
            is_private: bool = False,
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

        assert isinstance(name, str), "Attribute must have a name"
        assert isinstance(is_private, bool), "Attribute must be marked as private or public"

        # Validate the data type: int or float
        if data_type is None:
            try:
                data_type = type(data[0])
            except TypeError:
                data_type = type(data)

        if data_type in MeshAttribute.INTEGER_TYPE_ALIAS:
            data_type = MeshAttribute.INTEGER_TYPE
        elif data_type in MeshAttribute.FLOAT_TYPE_ALIAS:
            data_type = MeshAttribute.FLOAT_TYPE
        elif data_type is None:
            pass
        
        # Validate the data - fit into an array if it's a scalar
        if isinstance(data, (int, float, np.int, np.double)):
            pass
        elif isinstance(data, (list, set, tuple, np.array)):
            pass
        elif data is None:
            pass
        else:
            raise ValueError(f"`data` must be a number or a list, not: {type(data)}")
        
        # Validate that the attribute type is properly a node or cell attribute
        if attribute_type is None:
            pass
        elif attribute_type.lower() in MeshAttribute.NODE_TYPE_ALIAS:
            attribute_type = MeshAttribute.NODE_TYPE
        elif attribute_type.lower() in MeshAttribute.CELL_TYPE_ALIAS:
            attribute_type = MeshAttribute.CELL_TYPE
        else:
            raise ValueError(f"Attribute type must be node or cell, not: {attribute_type}")
        
        self.name = name
        self.attribute_type = attribute_type
        self.data = data
        self.data_type = data_type
        self.is_private = is_private
    
    @property
    def is_node_attribute(self):
        if self.attribute_type == MeshAttribute.NODE_TYPE:
            return True
        elif self.attribute_type == MeshAttribute.CELL_TYPE:
            return False
        else:
            raise TypeError(f"Attribute type is malformed: {self.attribute_type}")
    
    @property
    def is_cell_attribute(self):
        return not self.is_node_attribute
    
