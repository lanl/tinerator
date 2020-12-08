import numpy as np
import random
import os
from .mesh import load
from ..gis import map_elevation

def triangulation_jigsaw(raster, boundary_distance: float):
    '''
    Triangulates using the Python wrapper for JIGSAW.
    Author: Darren Engwirda.
    '''
    try:
        import jigsawpy
    except ModuleNotFoundError:
        err = "The `jigsawpy` module is not installed. "
        err += "Build directions can be found at:\n"
        err += "  https://github.com/dengwirda/jigsaw-python"
        raise ModuleNotFoundError(err)

    boundary = raster.get_boundary(distance=boundary_distance)

    verts = [((pt[0], pt[1]), 0) for pt in boundary.points]
    conn = [((i, i+1), 0) for i in range(len(boundary.points) - 1)]
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
    opts.optm_qlim = +.95

    opts.mesh_top1 = True
    opts.geom_feat = True

    jigsawpy.lib.jigsaw(opts, geom, mesh)

    scr2 = jigsawpy.triscr2(
        mesh.point["coord"],
        mesh.tria3["index"]
    )

    outfile = f"tmp_mesh_{int(random.random() * 100000)}.vtk"
    jigsawpy.savevtk(outfile, mesh)

    mesh = load(outfile, driver="vtk", block_id=1, name="jigsaw-triplane")

    try:
        os.remove(outfile)
    except:
        print(f"Warning: could not delete temporary file {outfile}.")

    mesh.nodes[:, 2] = map_elevation(raster, mesh.nodes)
    
    return mesh