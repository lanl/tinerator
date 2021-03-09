import numpy as np
from .mesh import Mesh
from ..logging import warn
from ..gis import Raster

def triangulate(
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
            from .triangulation_jigsaw import triangulation_jigsaw

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
