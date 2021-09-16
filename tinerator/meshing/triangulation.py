import numpy as np
from typing import Union, List
from .mesh import Mesh
from .triangulation_jigsaw import triangulation_jigsaw
from .triangulation_meshpy import triangulation_meshpy
from ..logging import warn, debug, log
from ..gis import Raster, Geometry

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
    refinement_feature: Union[Geometry, List[Geometry]] = None,
    method: str = "default",
    scaling_type: str = "relative",
    verbosity_level: int = 1,
    **kwargs,
) -> Mesh:
    """
    Generates a quality triangulation of a raster.

    The core of the triangulation algorithm are the parameters:

    - ``min_edge_length``: The minimum edge length for created triangles.
    - ``max_edge_length``: The maximum edge length for created triangles.
    - ``scaling_type``: Can be "relative" or "absolute". If absolute, then the edge
        length parameters will interpreted as meters (or whatever the units of
        the DEM are). If "relative", then the edge length parameters will be considered
        percent of the DEM extent.

    Recall that the refinement feature create a distance field internally. That distance
    field is normalized to [0., 1.].

    Where the distance field is 0., triangles will have an edge length of ``min_edge_length``.
    Where the distance field is 1., triangles will have an edge length of ``max_edge_length``.

    Distance field values between [0., 1.] will linearly interpolate the edge lengths.

    Available options for the ``method`` argument are:

    - ``"jigsaw"``: Uses `JIGSAW <https://github.com/dengwirda/jigsaw-python>`_ to generate a triangulation.
    - ``"meshpy"``: Uses `MeshPy <https://documen.tician.de/meshpy/index.html>`_ to generate a triangulation.
    - ``"gmsh"``: (in progress; not available yet). Uses `Gmsh <https://gmsh.info>`_ to generate a triangulation.

    The default is ``"meshpy"``.

    Args
    ----
        raster (Raster): The TINerator Raster object to generate a triangulation from.
        min_edge_length (:obj:`float`, optional): The minimum triangle edge length to generate.
        max_edge_length (:obj:`float`, optional): The maximum triangle edge length to generate.
        refinement_feature (:obj:`Geometry`, optional): The Geometry object used to refine the triangulation around.
        method (:obj:`str`, optional): The triangulation kernel to use. Defaults to ``"jigsaw"``.
        scaling_type (:obj:`str`, optional): Defines how to interpret the provided edge lengths, as "relative" or "absolute".
        verbosity_level (:obj:`int`, optional): The level of printed output on meshing progress. From 0 (none) to 3 (maximum).
        **kwargs: Optional parameters to pass into the triangulation kernel.

    Returns
    -------
        Mesh: A TINerator triangulation surface mesh.

    Examples
    --------
        >>> surf_mesh = tin.meshing.triangulate(dem, min_edge_length=0.001, scaling_type="relative")
    """

    if raster.units.lower() == "degree":
        warn(
            "IMPORTANT: The CRS of the raster is in degrees. "
            "This will cause problems with some triangulation kernels. "
            "Reproject into a meter- or feet- based CRS."
        )

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
    boundary = raster.get_boundary()

    return triang_cb(
        raster,
        raster_boundary=boundary,
        refinement_feature=refinement_feature,
        min_edge_length=min_edge_length,
        max_edge_length=max_edge_length,
        scaling_type=scaling_type,
        verbosity_level=verbosity_level,
        **kwargs,
    )
