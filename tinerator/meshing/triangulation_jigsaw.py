import numpy as np
import random
import os
import tempfile
from .mesh import load
from ..gis import map_elevation, Raster, distance_map
from ..logging import log, warn, debug


def triangulation_jigsaw_refined(
    raster, min_edge: float, max_edge: float, refinement_feature
):
    """
    Triangulates using the Python wrapper for JIGSAW.
    Author: Darren Engwirda.
    """
    try:
        import jigsawpy
    except ModuleNotFoundError:
        err = "The `jigsawpy` module is not installed. "
        err += "Build directions can be found at:\n"
        err += "  https://github.com/dengwirda/jigsaw-python"
        raise ModuleNotFoundError(err)

    src_path = "/Users/livingston/playground/lanl/tinerator/tinerator-core/tpl/jigsaw-python/files"
    dst_path = "/Users/livingston/playground/lanl/tinerator/tinerator-test-cases/tmp"

    opts = jigsawpy.jigsaw_jig_t()

    geom = jigsawpy.jigsaw_msh_t()
    mesh = jigsawpy.jigsaw_msh_t()
    hmat = jigsawpy.jigsaw_msh_t()

    # ===================== #

    log(f"Creating refined triplane with JIGSAW with edge lengths {min_edge} -> {max_edge}")

    boundary = raster.get_boundary(distance=min_edge)

    debug("Setting JIGSAW parameters")

    verts = [((pt[0], pt[1]), 0) for pt in boundary.points]
    conn = [((i, i + 1), 0) for i in range(len(boundary.points) - 1)]
    conn += [((len(boundary.points) - 1, 0), 0)]

    #geom = jigsawpy.jigsaw_msh_t()

    #geom.mshID = "euclidean-mesh"
    #geom.ndims = +2
    #geom.vert2 = np.array(verts, dtype=geom.VERT2_t)
    #geom.edge2 = np.array(conn, dtype=geom.EDGE2_t)

    ttttt = "/Users/livingston/playground/lanl/tinerator/tinerator-test-cases/tmp/testgeom.msh"

    with open(ttttt, "w") as f:
        f.write("mshid=1\n")
        f.write("ndims=2\n")
        f.write("point=%d\n" % len(verts))

        for p in verts:
            f.write(f"{p[0][0]};{p[0][1]};0\n")

        f.write("edge2=%d\n" % len(conn))

        for c in conn:
            f.write(f"{c[0][0]};{c[0][1]};0\n")

    # ===================== #

#------------------------------------ setup files for JIGSAW

    opts.geom_file = ttttt
    opts.jcfg_file = os.path.join(dst_path, "airfoil.jig")
    opts.mesh_file = os.path.join(dst_path, "airfoil.msh")
    opts.hfun_file = os.path.join(dst_path, "spacing.msh")

#------------------------------------ compute HFUN over GEOM

    jigsawpy.loadmsh(opts.geom_file, geom)

    xgeo = geom.vert2["coord"][:, 0]
    ygeo = geom.vert2["coord"][:, 1]

    xpos = np.linspace(xgeo.min(), xgeo.max(), 80)
    ypos = np.linspace(ygeo.min(), ygeo.max(), 40)

    xmat, ymat = np.meshgrid(xpos, ypos)

    #fun1 = +0.1 * (xmat - .40) ** 2 + +2.0 * (ymat - .55) ** 2
    #fun2 = +0.7 * (xmat - .75) ** 2 + +0.7 * (ymat - .45) ** 2
    #hfun = np.minimum(fun1, fun2)
    #hmin = 0.01; hmax = 0.10
    #hfun = 0.4 * np.maximum(np.minimum(hfun, hmax), hmin)

    dmap = distance_map(raster, refinement_feature)

    hmat.mshID = "euclidean-grid"
    hmat.ndims = +2
    hmat.xgrid = np.array(xpos, dtype=hmat.REALS_t)
    hmat.ygrid = np.array(ypos, dtype=hmat.REALS_t)
    hmat.value = np.array(dmap, dtype=hmat.REALS_t)

    jigsawpy.savevtk(os.path.join(dst_path, "case_5a_aaaaaa.vtk"), hmat)

    #from matplotlib import pyplot as plt; plt.imshow(dmap); plt.show(); exit(1)

#------------------------------------ make mesh using JIGSAW

    opts.hfun_scal = "absolute"
    opts.hfun_hmax = float("inf")       # null HFUN limits
    opts.hfun_hmin = float(+0.00)

    opts.mesh_kern = "delfront"         # DELFRONT kernel
    opts.mesh_dims = +2

    opts.geom_feat = True
    opts.mesh_top1 = True

    jigsawpy.cmd.jigsaw(opts, mesh)

    with tempfile.TemporaryDirectory() as tmp_dir:
        outfile = os.path.join(tmp_dir, "mesh.vtk")
        
        debug(f"Writing triangulation to disk: {outfile}")

        jigsawpy.savevtk(outfile, mesh)
        mesh = load(outfile, driver="vtk", block_id=1, name="jigsaw-triplane")

    #mesh.nodes[:, 2] = map_elevation(raster, mesh.nodes)

    return mesh


def triangulation_jigsaw_refined__backup(
    raster, min_edge: float, max_edge: float, refinement_feature
):
    """
    Triangulates using the Python wrapper for JIGSAW.
    Author: Darren Engwirda.
    """
    try:
        import jigsawpy
    except ModuleNotFoundError:
        err = "The `jigsawpy` module is not installed. "
        err += "Build directions can be found at:\n"
        err += "  https://github.com/dengwirda/jigsaw-python"
        raise ModuleNotFoundError(err)

    log(f"Creating refined triplane with JIGSAW with edge lengths {min_edge} -> {max_edge}")

    boundary = raster.get_boundary(distance=min_edge)

    debug("Setting JIGSAW parameters")

    verts = [((pt[0], pt[1]), 0) for pt in boundary.points]
    conn = [((i, i + 1), 0) for i in range(len(boundary.points) - 1)]
    conn += [((len(boundary.points) - 1, 0), 0)]

    opts = jigsawpy.jigsaw_jig_t()
    geom = jigsawpy.jigsaw_msh_t()
    mesh = jigsawpy.jigsaw_msh_t()
    hmat = jigsawpy.jigsaw_msh_t()

    geom.mshID = "euclidean-mesh"
    geom.ndims = +2
    geom.vert2 = np.array(verts, dtype=geom.VERT2_t)
    geom.edge2 = np.array(conn, dtype=geom.EDGE2_t)

    xgeo = geom.vert2["coord"][:, 0]
    ygeo = geom.vert2["coord"][:, 1]

    xpos = np.linspace(xgeo.min(), xgeo.max(), 80)

    ypos = np.linspace(ygeo.min(), ygeo.max(), 40)

    xmat, ymat = np.meshgrid(xpos, ypos)

    fun1 = +0.1 * (xmat - 0.40) ** 2 + +2.0 * (ymat - 0.55) ** 2

    fun2 = +0.7 * (xmat - 0.75) ** 2 + +0.7 * (ymat - 0.45) ** 2

    hfun = np.minimum(fun1, fun2)

    hmin = 0.01
    hmax = 0.10

    hfun = 0.4 * np.maximum(np.minimum(hfun, hmax), hmin)

    hmat.mshID = "euclidean-grid"
    hmat.ndims = +2
    hmat.xgrid = np.array(xpos, dtype=hmat.REALS_t)
    hmat.ygrid = np.array(ypos, dtype=hmat.REALS_t)
    hmat.value = np.array(hfun, dtype=hmat.REALS_t)

    # raise ValueError("asdf")

    # opts.hfun_hmax = 0.05
    # opts.mesh_dims = +2
    # opts.optm_qlim = +.95

    # opts.mesh_top1 = True
    # opts.geom_feat = True

    # jigsawpy.lib.jigsaw(opts, geom, mesh)

    # scr2 = jigsawpy.triscr2(
    #    mesh.point["coord"],
    #    mesh.tria3["index"]
    # )

    # outfile = f"tmp_mesh_{int(random.random() * 100000)}.vtk"
    # jigsawpy.savevtk(outfile, mesh)

    opts.hfun_scal = "absolute"
    opts.hfun_hmax = float("inf")  # null HFUN limits
    opts.hfun_hmin = float(+0.00)

    opts.mesh_kern = "delfront"  # DELFRONT kernel
    opts.mesh_dims = +2

    opts.geom_feat = True
    opts.mesh_top1 = True

    jigsawpy.cmd.jigsaw(opts, mesh)

    mesh = load(outfile, driver="vtk", block_id=1, name="jigsaw-triplane")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = "/Users/livingston/playground/lanl/tinerator/tinerator-test-cases/tmp"
        outfile = os.path.join(tmp_dir, "mesh.vtk")
        
        debug(f"Writing triangulation to disk: {outfile}")

        jigsawpy.savevtk(outfile, mesh)
        mesh = load(outfile, driver="vtk", block_id=1, name="jigsaw-triplane")
    
    mesh.nodes[:, 2] = map_elevation(raster, mesh.nodes)

    return mesh

    # ========== #

    opts = jigsawpy.jigsaw_jig_t()

    geom = jigsawpy.jigsaw_msh_t()
    mesh = jigsawpy.jigsaw_msh_t()
    hmat = jigsawpy.jigsaw_msh_t()

    xgeo = geom.vert2["coord"][:, 0]
    ygeo = geom.vert2["coord"][:, 1]

    xpos = np.linspace(xgeo.min(), xgeo.max(), 80)

    ypos = np.linspace(ygeo.min(), ygeo.max(), 40)

    xmat, ymat = np.meshgrid(xpos, ypos)

    fun1 = +0.1 * (xmat - 0.40) ** 2 + +2.0 * (ymat - 0.55) ** 2

    fun2 = +0.7 * (xmat - 0.75) ** 2 + +0.7 * (ymat - 0.45) ** 2

    hfun = np.minimum(fun1, fun2)

    hmin = 0.01
    hmax = 0.10

    hfun = 0.4 * np.maximum(np.minimum(hfun, hmax), hmin)

    hmat.mshID = "euclidean-grid"
    hmat.ndims = +2
    hmat.xgrid = np.array(xpos, dtype=hmat.REALS_t)
    hmat.ygrid = np.array(ypos, dtype=hmat.REALS_t)
    hmat.value = np.array(hfun, dtype=hmat.REALS_t)

    jigsawpy.savemsh(opts.hfun_file, hmat)

    # ------------------------------------ make mesh using JIGSAW

    opts.hfun_scal = "absolute"
    opts.hfun_hmax = float("inf")  # null HFUN limits
    opts.hfun_hmin = float(+0.00)

    opts.mesh_kern = "delfront"  # DELFRONT kernel
    opts.mesh_dims = +2

    opts.geom_feat = True
    opts.mesh_top1 = True

    jigsawpy.cmd.jigsaw(opts, mesh)

    print("Saving case_5a.vtk file.")

    jigsawpy.savevtk(os.path.join(dst_path, "case_5a.vtk"), hmat)

    print("Saving case_5b.vtk file.")

    jigsawpy.savevtk(os.path.join(dst_path, "case_5b.vtk"), mesh)


def triangulation_jigsaw(
        raster: Raster, 
        boundary_distance: float,
    ):
    """
    Triangulates using the Python wrapper for JIGSAW.
    Author: Darren Engwirda.
    """

    #init_near = 1.e-6
    #geom_seed = 16
    #geom_feat = True
    #geom_eta1 = 60
    #geom_eta2 = 60
    ## hfun_file = *.msh
    
    #hfun_scal = "relative"; ["relative", "absolute"]
    #hfun_max = 0.02
    #hfun_min = 0.00

    ## mesh_dims = 2
    #mesh_kern = "delaunay"; ["delaunay", "delfront"]
    #mesh_iter = 1e6
    #mesh_top1 = False
    #mesh_top2 = True
    #mesh_rad2 = 1.05
    #mesh_rad3 = 2.05
    #mesh_off2 = 0.90
    #mesh_off3 = 1.10
    #mesh_snk2 = 0.25
    #mesh_snk3 = 0.33
    #mesh_eps1 = 0.33
    #mesh_eps2 = 0.33
    #mesh_vol3 = 0.10

    #optm_kern = "odt+dqdx"; ["odt+dqdx", "cvt+dqdx"]
    #optm_iter = 16
    #optm_qtol = 1.e-4
    #optm_qlim = 0.9250
    #optm_zip_ = True
    #optm_dv_ = True
    #optm_tria = True
    #optm_dual = True

    #verbosity = 0


    # https://github.com/dengwirda/jigsaw/blob/master/example.jig

    try:
        import jigsawpy
    except ModuleNotFoundError:
        err = "The `jigsawpy` module is not installed. "
        err += "Build directions can be found at:\n"
        err += "  https://github.com/dengwirda/jigsaw-python"
        raise ModuleNotFoundError(err)

    log(f"Creating uniform triplane with JIGSAW at edge length {boundary_distance}")

    boundary = raster.get_boundary(distance=boundary_distance)

    debug("Setting JIGSAW parameters")

    verts = [((pt[0], pt[1]), 0) for pt in boundary.points]
    conn = [((i, i + 1), 0) for i in range(len(boundary.points) - 1)]
    conn += [((len(boundary.points) - 1, 0), 0)]

    opts = jigsawpy.jigsaw_jig_t()
    geom = jigsawpy.jigsaw_msh_t()
    mesh = jigsawpy.jigsaw_msh_t()

    geom.mshID = "euclidean-mesh"
    geom.ndims = +2
    geom.vert2 = np.array(verts, dtype=geom.VERT2_t)
    geom.edge2 = np.array(conn, dtype=geom.EDGE2_t)

    # max. mesh-size function value. Interpreted based on SCAL setting.
    opts.hfun_scal = "absolute" # relative
    opts.hfun_hmax = 100. #0.05

    # --> max edge = 
    # xmin, ymin, xmax, ymax = dem.extent
    # hfun_hmax * np.mean(((ymax - ymin), (xmax - xmin)))

    # number of "topological" dimensions to mesh.
    opts.mesh_dims = +2

    # threshold on mesh cost function above which 
    # gradient-based optimisation is attempted.
    opts.optm_qlim = +0.95

    # enforce 1-dim. topological constraints.
    opts.mesh_top1 = True

    # attempt to auto-detect "sharp-features" in the input geometry.
    opts.geom_feat = True

    jigsawpy.lib.jigsaw(opts, geom, mesh)

    debug("Starting triangulation")

    scr2 = jigsawpy.triscr2(mesh.point["coord"], mesh.tria3["index"])

    with tempfile.TemporaryDirectory() as tmp_dir:
        outfile = os.path.join(tmp_dir, "mesh.vtk")
        
        debug(f"Writing triangulation to disk: {outfile}")

        jigsawpy.savevtk(outfile, mesh)
        mesh = load(outfile, driver="vtk", block_id=1, name="jigsaw-triplane")

    mesh.nodes[:, 2] = map_elevation(raster, mesh.nodes)

    return mesh
