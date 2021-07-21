import numpy as np
import random
import os
import tempfile
from .mesh import load_mesh
from .meshing_utils import get_linestring_connectivity
from ..gis import map_elevation, Raster, Geometry, distance_map
from ..logging import log, warn, debug

"""
JIGSAW_DEFAULTS = {
    #------------------------------------------ MISC options
    verbosity = None

        self.jcfg_file = None

        self.tria_file = None
        self.bnds_file = None

    #------------------------------------------ INIT options
        self.init_file = None
        self.init_tags = None
        self.init_near = None

    #------------------------------------------ GEOM options
        self.geom_file = None
        self.geom_tags = None

        self.geom_seed = None

        self.geom_feat = None

        self.geom_phi1 = None
        self.geom_phi2 = None

        self.geom_eta1 = None
        self.geom_eta2 = None

    #------------------------------------------ HFUN options
        self.hfun_file = None
        self.hfun_tags = None

        self.hfun_scal = None

        self.hfun_hmax = None
        self.hfun_hmin = None

    #------------------------------------------ MESH options
        self.mesh_file = None
        self.mesh_tags = None

        self.mesh_kern = None
        self.bnds_kern = None

        self.mesh_iter = None

        self.mesh_dims = None

        self.mesh_top1 = None
        self.mesh_top2 = None

        self.mesh_siz1 = None
        self.mesh_siz2 = None
        self.mesh_siz3 = None

        self.mesh_eps1 = None
        self.mesh_eps2 = None

        self.mesh_rad2 = None
        self.mesh_rad3 = None

        self.mesh_off2 = None
        self.mesh_off3 = None

        self.mesh_snk2 = None
        self.mesh_snk3 = None

        self.mesh_vol3 = None

    #------------------------------------------ OPTM options
        self.optm_kern = None

        self.optm_iter = None

        self.optm_qtol = None
        self.optm_qlim = None

        self.optm_zip_ = None
        self.optm_div_ = None
        self.optm_tria = None
        self.optm_dual = None
}
"""

def triangulation_jigsaw(
    raster: Raster,
    raster_boundary: Geometry = None,
    min_edge_length: float = None,
    max_edge_length: float = None,
    refinement_feature: Geometry = None,
    scaling_type: str = "relative",
    meshing_kernel: str = "delfront",
    **kwargs
):
    """
    Uses JIGSAW to triangulate a raster.

    Parameters
    ==========

    raster : TINerator Raster object
    raster_boundary : Geometry
    min_edge_length : float
    max_edge_length : float, optional
    refinement_feature : Geometry, optional
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

    log("Triangulating raster with JIGSAW")

    # TODO: does this respect nodes that aren't already ordered
    # clockwise?
    # vertices, segments = raster_boundary
    vertices = np.array(raster_boundary.shapes[0].coords[:])
    segments = get_linestring_connectivity(vertices, closed=True, clockwise=True)

    verts = [((pt[0], pt[1]), 0) for pt in vertices]
    conn = [((i, i + 1), 0) for i in range(len(vertices) - 1)]
    conn += [((len(vertices) - 1, 0), 0)]

    opts = jigsawpy.jigsaw_jig_t()
    geom = jigsawpy.jigsaw_msh_t()
    mesh = jigsawpy.jigsaw_msh_t()
    hmat = None

    # Construct geometry object
    geom.mshID = "euclidean-mesh"
    geom.ndims = +2
    geom.vert2 = np.array(verts, dtype=geom.VERT2_t)
    geom.edge2 = np.array(conn, dtype=geom.EDGE2_t)

    if False:
        from matplotlib import pyplot as plt
        print("====")
        print(geom.vert2)
        print(geom.edge2)

        dbg_verts = np.array([x for x, _ in verts])
        dbg_edges = np.array([x for x, _ in conn])
        print(dbg_verts[dbg_edges])

        print("====")
        f = plt.figure()

        for x in dbg_verts[dbg_edges]:
            dist = np.linalg.norm(x[0] - x[1])
            print(dist)
            plt.plot(x[:,0], x[:,1])
        plt.scatter(dbg_verts[:,0], dbg_verts[:,1], c=list(range(len(verts))))
        plt.savefig("DEBUG-verts.png")
        exit()

    # Construct JIGSAW opts object
    opts.hfun_scal = scaling_type  # Interpret HMIN/HMAX as relative or absolute?
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
            max_dist=max_edge_length,
        )

        hmat = jigsawpy.jigsaw_msh_t()
        hmat.mshID = "euclidean-grid"
        hmat.ndims = +2

        xmin, ymin, xmax, ymax = dmap.extent

        xpos = np.linspace(xmin, xmax, dmap.ncols)
        ypos = np.linspace(ymin, ymax, dmap.nrows)

        # The dmap array needs to be flipped vertically
        hfunc = np.flipud(np.array(dmap.data, dtype=float))

        hmat.xgrid = np.array(xpos, dtype=hmat.REALS_t)
        hmat.ygrid = np.array(ypos, dtype=hmat.REALS_t)
        hmat.value = np.array(hfunc, dtype=hmat.REALS_t)

        # Set to 0 and +inf:
        # HMIN and HMAX are contained within the HMAT raster now
        opts.hfun_hmin = float(+0.00)
        opts.hfun_hmax = float("inf")

    # Run the triangulation algorithm
    debug("Beginning triangulation...")
    jigsawpy.lib.jigsaw(opts, geom, mesh, hfun=hmat)
    debug("Finished triangulation")

    with tempfile.TemporaryDirectory() as tmp_dir:
        outfile = os.path.join(tmp_dir, "mesh.vtk")

        debug(f"Writing triangulation to disk: {outfile}")

        jigsawpy.savevtk(outfile, mesh)
        mesh = load_mesh(outfile, driver="vtk", block_id=1, name="jigsaw-triplane")

    mesh.nodes[:, 2] = map_elevation(raster, mesh.nodes)
    mesh.crs = raster_boundary.crs
    mesh.material_id = 1

    return mesh
