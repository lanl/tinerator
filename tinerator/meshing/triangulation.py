import numpy as np
from .mesh import Mesh
from .triangulation_jigsaw import triangulation_jigsaw
from .triangulation_meshpy import triangulation_meshpy
from ..logging import warn, debug, log
from ..gis import Raster

TRIANGULATION_METHODS = {
    "default": triangulation_meshpy,
    "meshpy": triangulation_meshpy,
    "jigsaw": triangulation_jigsaw,
}

# from .wavelet_lg import triangulation_wavelet
# from .refined_triplane_lg import build_refined_triplane
# from .uniform_triplane_lg import build_uniform_triplane
# from .triangulation_poisson_disc_sampling import triangulation_poisson_disc_sampling


def triangulate(
    raster: Raster,
    min_edge_length: float = None,
    max_edge_length: float = None,
    refinement_feature=None,
    method="default",
    scaling_type="relative",
    **kwargs,
):
    """
    Triangulate a raster.
    """

    valid_callbacks = list(TRIANGULATION_METHODS.keys())

    if method not in valid_callbacks:
        raise ValueError(
            f'Incorrect method "{method}". Must be one of: {valid_callbacks}'
        )

    triang_cb = TRIANGULATION_METHODS[method]

    if scaling_type == "relative":
        xmin, ymin, xmax, ymax = raster.extent
        coeff = min_edge_length if min_edge_length is not None else max_edge_length
        boundary_dist = coeff * np.mean((np.abs(ymax - ymin), np.abs(xmax - xmin)))
    elif scaling_type == "absolute":
        boundary_dist = (
            min_edge_length if min_edge_length is not None else max_edge_length
        )
    else:
        raise ValueError(
            f'Incorrect scaling type "{scaling_type}". Must be one of: ["relative", "absolute"].'
        )

    debug(f"Generating boundary for triangulation at distance = {boundary_dist}")
    boundary = raster.get_boundary(distance=boundary_dist)

    return triang_cb(
        raster,
        raster_boundary=boundary,
        refinement_feature=refinement_feature,
        min_edge_length=min_edge_length,
        max_edge_length=max_edge_length,
        scaling_type=scaling_type,
        **kwargs,
    )
