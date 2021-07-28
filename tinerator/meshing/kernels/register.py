_REGISTERED = {
    "triangle": [],
    "quad": [],
    "tetrahedral": [],
    "hex": [],
    "polygon": [],
    "mixed": []
}

def register_kernel(
    schema: dict,  
):
    """
    """

    try:
        _REGISTERED[format["metadata"]["output_mesh"]].append(schema)
    except KeyError as e:
        raise KeyError(f"Malformed or unsupported element type. {e}")

def get_registered_kernels():

    names = {}

    for key in keys(_REGISTERED):
        if len(_REGISTERED[key]) > 0:
            names[key] = [(x["metadata"]["name"], x["metadata"]["description"]) for x in _REGISTERED[key]]

    return names