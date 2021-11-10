from copy import deepcopy
from .lagrit_helper import write_line, cleanup
from .mesh import read_avs
from ..gis import Raster, DistanceMap
from ..gis import map_elevation


def build_uniform_triplane(
    dem_raster: Raster, edge_length: float, verbose: bool = False
):
    # The algorithm works by triangulating only the boundary, and
    # then iterating through each triangle, breaking edges in half
    # where they exceed the given edge length.
    # Consequently, this means that the final length scale will have
    # a minimum edge length of `edge_length / 2`, and a maximum edge
    # length of `edge_length`.

    edge_length = edge_length * 2.0

    from pylagrit import PyLaGriT

    lg = PyLaGriT(verbose=verbose)

    boundary = dem_raster.get_boundary(distance=edge_length)
    counterclockwise = False

    # Generate the boundary polygon
    write_line(boundary.points, "poly_1.inp", connections=boundary.connectivity)

    # Compute length scales to break triangles down into
    # See below for a more in-depth explanation
    length_scales = [edge_length * i for i in [1, 2, 4, 8, 16, 32, 64]][::-1]

    mo_tmp = lg.read("poly_1.inp")

    motri = lg.create(elem_type="triplane")
    motri.setatt("ipolydat", "no")
    lg.sendline("copypts / %s / %s" % (motri.name, mo_tmp.name))
    motri.setatt("imt", 1)
    mo_tmp.delete()

    # Triangulate the boundary
    motri.select()
    if counterclockwise:
        motri.triangulate(order="counterclockwise")
    else:
        motri.triangulate(order="clockwise")

    # Set material ID to 1
    motri.setatt("itetclr", 1)
    motri.setatt("motri", 1)
    motri.resetpts_itp()

    lg.sendline("cmo/copy/mo/%s" % motri.name)

    # Move through each length scale, breaking down edges less than the value 'ln'
    # Eventually this converges on triangles with edges in
    # the range [0.5*edge_length,edge_length]
    motri.select()
    for ln in length_scales:

        motri.refine(
            refine_option="rivara",
            refine_type="edge",
            values=[ln],
            inclusive_flag="inclusive",
        )

        for _ in range(3):
            motri.recon(0)
            motri.smooth()
        motri.rmpoint_compress(resetpts_itp=False)

    # Smooth and reconnect the triangulation
    for _ in range(6):
        motri.smooth()
        motri.recon(0)
        motri.rmpoint_compress(resetpts_itp=True)

    motri.rmpoint_compress(resetpts_itp=False)
    motri.recon(1)
    motri.smooth()
    motri.recon(0)
    motri.recon(1)

    motri.setatt("ipolydat", "yes")
    motri.dump("surface_lg.inp")

    # TODO: change to Exodus!
    mo = read_avs(
        "surface_lg.inp",
        keep_material_id=False,
        keep_node_attributes=False,
        keep_cell_attributes=False,
    )

    cleanup(["poly_1.inp", "surface_lg.inp", "lagrit.log", "lagrit.out"])

    z_values = map_elevation(dem_raster, mo.nodes)
    mo.nodes[:, 2] = z_values

    return mo
