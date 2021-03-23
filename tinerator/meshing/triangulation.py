import numpy as np
from .mesh import Mesh
from .triangulation_jigsaw import triangulation_jigsaw, triangulation_jigsaw_refined
from ..logging import warn, debug, log
from ..gis import Raster

TRIANGULATION_METHODS = {
    "jigsaw": triangulation_jigsaw,
    "jigsaw2": triangulation_jigsaw_refined,
}


def triangulate(
    raster: Raster,
    min_edge_length: float = None,
    max_edge_length: float = None,
    refinement_feature = None,
    method = "jigsaw",
    scaling_type = "relative",
    **kwargs
):
    """
    Triangulate a raster.
    """

    valid_callbacks = list(TRIANGULATION_METHODS.keys())

    if method not in valid_callbacks:
        raise ValueError(f"Incorrect method \"{method}\". Must be one of: {valid_callbacks}")

    triang_cb = TRIANGULATION_METHODS[method]

    if scaling_type == "relative":
        xmin, ymin, xmax, ymax = raster.extent
        coeff = min_edge_length if min_edge_length is not None else max_edge_length
        boundary_dist = coeff * np.mean((np.abs(ymax - ymin), np.abs(xmax - xmin)))
    elif scaling_type == "absolute":
        boundary_dist = min_edge_length if min_edge_length is not None else max_edge_length
    else:
        raise ValueError(f"Incorrect scaling type \"{scaling_type}\". Must be one of: [\"relative\", \"absolute\"].")

    debug(f"Generating boundary for triangulation at distance = {boundary_dist}")
    boundary = raster.get_boundary(distance=boundary_dist)

    return triang_cb(raster, raster_boundary=boundary, refinement_feature=refinement_feature, min_edge_length=min_edge_length, max_edge_length=max_edge_length, scaling_type=scaling_type, **kwargs)

def triangulate_old(
    raster: Raster,
    min_edge: float,
    max_edge: float = None,
    refinement_feature=None,
    method: str = "default",
    **kwargs,
) -> Mesh:
    """
    Generates a triangular surface mesh (triplane) from a Raster object.
    """

    uniform_edges = True if max_edge is None else False

    method = method.lower().strip()

    if uniform_edges:
        if refinement_feature is not None:
            warn(
                "`max_edge` undefined but `refinement_feature` was provided."
                + "\n"
                + "Generating uniform mesh and `refinement_feature` will be ignored."
            )

        if method == "default":
            from .triangulation_triangle import triangulation_triangle_uniform

            return triangulation_triangle_uniform(raster, min_edge, **kwargs)
        elif method == "lagrit":
            from .uniform_triplane_lg import build_uniform_triplane

            return build_uniform_triplane(raster, min_edge, **kwargs)
        elif method == "jigsaw":
            

            return triangulation_jigsaw(raster, min_edge, **kwargs)
        else:
            raise ValueError(
                f'Unknown uniform triangulation method: "{method}"'
            )
    else:

        if refinement_feature is None:
            raise ValueError("For refined triangulation, `refinement_feature` is required")

        if method == "lagrit":
            from .refined_triplane_lg import build_refined_triplane

            return build_refined_triplane(
                raster, refinement_feature, min_edge, max_edge, **kwargs
            )
        elif method == "jigsaw":
            from .triangulation_jigsaw import triangulation_jigsaw_refined

            return triangulation_jigsaw_refined(
                raster, min_edge, max_edge, refinement_feature, **kwargs
            )
        else:
            raise ValueError(
                f'Unknown refined triangulation method: "{method}"'
            )
