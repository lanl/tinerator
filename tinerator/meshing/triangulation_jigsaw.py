import numpy as np
import random
import os
import tempfile
from .mesh import load
from ..gis import map_elevation, Raster, Shape, distance_map
from ..logging import log, warn, debug

def triangulation_jigsaw(
        raster: Raster, 
        raster_boundary: Shape = None,
        min_edge_length: float = None,
        max_edge_length: float = None,
        refinement_feature: Shape = None,
        scaling_type: str = "relative",
        meshing_kernel: str = "delfront",
        jigsaw_opts: dict = { 'verbosity': 1 },
    ):
    """
    Uses JIGSAW to triangulate a raster.

    Parameters
    ==========

    raster : TINerator Raster object
    raster_boundary : Shape
    min_edge_length : float
    max_edge_length : float, optional
    refinement_feature : Shape, optional
    scaling_type : {'absolute', 'relative'}, optional

      Scaling type for mesh-size function. `scaling_type = 'relative'`
      interprets mesh-size values as percentages of the (mean)
      length of the axis-aligned bounding-box (AABB) associated 
      with the geometry. `scaling_type = 'absolute'` interprets 
      mesh-size values as absolute measures.

    meshing_kernel : {'delfront', 'delaunay'}, optional
    jigsaw_opts: dictionary

    Returns
    =======

    A TINerator Mesh object.
    """

    assert scaling_type in ["absolute", "relative"]
    assert meshing_kernel in ["delfront", "delaunay"]

    try:
        import jigsawpy
    except ModuleNotFoundError:
        err = "The `jigsawpy` module is not installed. "
        err += "Build directions can be found at:\n"
        err += "  https://github.com/dengwirda/jigsaw-python"
        raise ModuleNotFoundError(err)

    log(f"Triangulating raster with JIGSAW")

    # TODO: does this respect nodes that aren't already ordered
    # clockwise?
    verts = [((pt[0], pt[1]), 0) for pt in raster_boundary.points]
    conn = [((i, i + 1), 0) for i in range(len(raster_boundary.points) - 1)]
    conn += [((len(raster_boundary.points) - 1, 0), 0)]

    opts = jigsawpy.jigsaw_jig_t()
    geom = jigsawpy.jigsaw_msh_t()
    mesh = jigsawpy.jigsaw_msh_t()
    hmat = None

    # Construct geometry object
    geom.mshID = "euclidean-mesh"
    geom.ndims = +2
    geom.vert2 = np.array(verts, dtype=geom.VERT2_t)
    geom.edge2 = np.array(conn, dtype=geom.EDGE2_t)

    # Construct JIGSAW opts object
    opts.hfun_scal = scaling_type # Interpret HMIN/HMAX as relative or absolute?
    opts.hfun_hmin = min_edge_length
    opts.hfun_hmax = max_edge_length
    opts.mesh_kern = meshing_kernel
    opts.mesh_dims = +2
    opts.mesh_top1 = True
    opts.geom_feat = True
    opts.optm_qlim = +0.95

    # Construct HMAT object (refinement function)
    if refinement_feature is not None:
        dmap = distance_map(
            raster, 
            refinement_feature, 
            min_dist=min_edge_length, 
            max_dist=max_edge_length
        )

        hmat = jigsawpy.jigsaw_msh_t()
        hmat.mshID = "euclidean-grid"
        hmat.ndims = +2

        dY, dX = dmap.shape
        xmin, ymin, xmax, ymax = raster.extent
        xpos = np.linspace(xmin, xmax, dX)
        ypos = np.linspace(ymin, ymax, dY)

        hmat.xgrid = np.array(xpos, dtype=hmat.REALS_t)
        hmat.ygrid = np.array(ypos, dtype=hmat.REALS_t)
        hmat.value = np.array(dmap, dtype=hmat.REALS_t)

        # Set to 0 and +inf: 
        # HMIN and HMAX are contained within the HMAT raster now
        opts.hfun_hmin = float(+0.00)
        opts.hfun_hmax = float("inf")

    # Allow for user-defined overrides of `opts`
    if jigsaw_opts is not None:
        for key in jigsaw_opts.keys():
            opts.__dict__[key] = jigsaw_opts[key]

    # Run the triangulation algorithm
    debug("Beginning triangulation...")
    jigsawpy.lib.jigsaw(opts, geom, mesh, hfun=hmat)
    debug("Finished triangulation")

    scr2 = jigsawpy.triscr2(mesh.point["coord"], mesh.tria3["index"])

    with tempfile.TemporaryDirectory() as tmp_dir:
        outfile = os.path.join(tmp_dir, "mesh.vtk")
        
        debug(f"Writing triangulation to disk: {outfile}")

        jigsawpy.savevtk(outfile, mesh)
        mesh = load(outfile, driver="vtk", block_id=1, name="jigsaw-triplane")

    mesh.nodes[:, 2] = map_elevation(raster, mesh.nodes)

    return mesh
