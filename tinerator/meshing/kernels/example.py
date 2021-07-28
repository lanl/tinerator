try:
    import blah
    from .register import register_kernel
    register_kernel(**return_registration())
except ModuleNotFoundError as e:
    print("blah blah blah")


def algo():
    pass


def return_registration():
    return {
        "function": algo,
        # triangle, quad, polygon, mixed
        "elements": ["triangle"],
        # distance_field, polyline, None
        "refinement": {
            "type": "distance_field",

        },
        "mesh": {
            "elements": ["triangle"],
            "format": "tinerator", # {"nodes": ..., "elements": ["type": ..., "connectivity": ...], "attributes": {"node": ..., "element": ...}}
            "has_Z": True,
        },
    }