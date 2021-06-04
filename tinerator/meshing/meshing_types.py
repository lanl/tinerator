import numpy as np
from enum import Enum, auto

class ElementType(Enum):
    TRIANGLE = auto()
    QUAD = auto()
    PRISM = auto()
    HEX = auto()
    POLYGON = auto()


AVS_TYPE_MAPPING = {"tri": ElementType.TRIANGLE, "prism": ElementType.PRISM}