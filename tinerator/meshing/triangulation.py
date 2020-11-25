import numpy as np
from .mesh import Mesh


def triangulate(
    raster,
    min_edge: float,
    max_edge: float = None,
    refinement_feature=None,
    method: str = "lagrit",
    **kwargs,
) -> Mesh:
    """
    Generates a triangular surface mesh (triplane) from a Raster object.
    """

    if max_edge is None:
        uniform_edges = True
    else:
        uniform_edges = False

    method = method.lower().strip()

    if uniform_edges:
        if refinement_feature is not None:
            print(
                "WARNING: `max_edge` undefined but `refinement_feature` was provided."
                + "\n"
                + "Generating uniform mesh and `refinement_feature` will be ignored."
            )

        if method == "lagrit":
            from .uniform_triplane_lg import build_uniform_triplane

            return build_uniform_triplane(raster, min_edge, **kwargs)
        else:
            raise ValueError(f"Unknown uniform triangulation method: {method}")
    else:
        if method == "lagrit":
            from .refined_triplane_lg import build_refined_triplane

            return build_refined_triplane(
                raster, refinement_feature, min_edge, max_edge, **kwargs
            )
        else:
            raise ValueError(f"Unknown refined triangulation method: {method}")
