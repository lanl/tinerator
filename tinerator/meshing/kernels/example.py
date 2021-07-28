import NonExistantModule


def algo():
    return {
        "nodes": [1,2,3],
        "elements": [
            {
                "type": "triangle",
                "connectivity": [[0,1,2], [2,4,2]]
            }
        ],
        "attributes": {
            "node": {},
            "element": {
                "material_id": [1,2,3,4]
            }
        }
    }


def register():
    return {
        "metadata": {
            "name": "JIGSAW",
            "description": algo.__doc__,
            "function": algo,
            "output_mesh": "triangle",
        },
        "input": {
            "clockwise_boundary": True,
            "refinement_type": "distance_field", # distance_field, polyline, None
        },
        "output": {
            "interpolate_Z": True,
        },
    }