import numpy as np
import random
import os
from .mesh import load
from ..gis import map_elevation


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

    boundary = raster.get_boundary(distance=min_edge)

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

    jigsawpy.savevtk("case_5a_test_ok.vtk", hmat)

    raise ValueError("asdf")

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

    try:
        os.remove(outfile)
    except:
        print(f"Warning: could not delete temporary file {outfile}.")

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


def triangulation_jigsaw(raster, boundary_distance: float):
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

    boundary = raster.get_boundary(distance=boundary_distance)

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

    opts.hfun_hmax = 0.05
    opts.mesh_dims = +2
    opts.optm_qlim = +0.95

    opts.mesh_top1 = True
    opts.geom_feat = True

    jigsawpy.lib.jigsaw(opts, geom, mesh)

    scr2 = jigsawpy.triscr2(mesh.point["coord"], mesh.tria3["index"])

    outfile = f"tmp_mesh_{int(random.random() * 100000)}.vtk"
    jigsawpy.savevtk(outfile, mesh)

    mesh = load(outfile, driver="vtk", block_id=1, name="jigsaw-triplane")

    try:
        os.remove(outfile)
    except:
        print(f"Warning: could not delete temporary file {outfile}.")

    mesh.nodes[:, 2] = map_elevation(raster, mesh.nodes)

    return mesh
