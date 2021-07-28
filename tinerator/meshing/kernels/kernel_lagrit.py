from copy import deepcopy
from .mesh import read_avs
from ..gis import Raster, Shape, DistanceMap
from ..gis import map_elevation

import os
import numpy as np


def filter_points(points: np.ndarray, eps: float):
    """
    Removes points that are within `eps` distance of each other.

    # Arguments
    points (np.ndarray): point array to filter
    eps (float): remove adjacent points within this distance of each other

    # Returns
    Filtered points
    """
    from scipy.spatial.distance import cdist

    mask = np.ones(np.shape(points)[0], dtype=bool)

    for (i, p) in enumerate(points):
        if mask[i]:
            dst = cdist(points, [p])
            mask[np.argwhere((dst > 0.0) & (dst < eps))] = False

    return points[mask]


def cleanup(files, silent=True):
    if not isinstance(files, list):
        files = [files]

    for file in files:
        try:
            os.remove(file)
        except Exception as e:
            if not silent:
                print(f"Could not remove {file}: {e.message}")


def line_connectivity(nodes: np.ndarray, connect_ends: bool = False):
    """
    Simple function to define a closed or open polyline for a set of
    nodes. Assumes adjacency in array == implicit connection.
    That is, requires a clockwise- or counter-clockwise set of nodes.
    """

    delta = 0 if connect_ends else -1
    size = np.shape(nodes)[0]
    connectivity = np.empty((size + delta, 2), dtype=np.int)
    for i in range(size - 1):
        connectivity[i] = np.array((i + 1, i + 2))
    if connect_ends:
        connectivity[-1] = np.array((size, 1))
    return connectivity


def write_line(
    boundary,
    outfile: str,
    connections=None,
    material_id=None,
    node_atts: dict = None,
    cell_atts: dict = None,
):

    nnodes = np.shape(boundary)[0]
    nlines = np.shape(connections)[0] if connections is not None else 0
    natts = len(node_atts.keys()) if node_atts is not None else 0
    catts = len(cell_atts.keys()) if cell_atts is not None else 0

    if material_id is not None:
        assert (
            np.shape(material_id)[0] >= nlines
        ), "Mismatch count between material ID and cells"

    with open(outfile, "w") as f:
        f.write("{} {} {} {} 0\n".format(nnodes, nlines, natts, catts))

        for i in range(nnodes):
            f.write("{} {} {} 0.0\n".format(i + 1, boundary[i][0], boundary[i][1]))

        for i in range(nlines):
            mat_id = material_id[i] if material_id is not None else 1
            f.write(
                "{} {} line {} {}\n".format(
                    i + 1, mat_id, connections[i][0], connections[i][1]
                )
            )

        if natts:

            for key in node_atts.keys():
                assert np.shape(node_atts[key])[0] >= nnodes, (
                    "Length of node attribute %s does not match length of points array"
                    % key
                )

            # 00007  1  1  1  1  1  1  1
            f.write(str(natts) + " 1" * natts + "\n")

            # imt1, integer
            # itp1, integer
            _t = "\n".join(
                [
                    key + ", " + "integer" if node_atts[key].dtype == int else "real"
                    for key in node_atts.keys()
                ]
            )
            f.write(_t + "\n")

            for i in range(nnodes):
                _att_str = "%d" % (i + 1)
                for key in node_atts.keys():
                    _att_str += " " + str(node_atts[key][i])
                _att_str += "\n"
                f.write(_att_str)

        if catts:

            for key in cell_atts.keys():
                assert np.shape(cell_atts[key])[0] >= nlines, (
                    "Length of cell attribute %s does not match length of elem array"
                    % key
                )

            # 00007  1  1  1  1  1  1  1
            f.write(str(catts) + " 1" * catts + "\n")

            # imt1, integer
            # itp1, integer
            _t = "\n".join(
                [
                    key + ", " + "integer" if cell_atts[key].dtype == int else "real"
                    for key in cell_atts.keys()
                ]
            )
            f.write(_t + "\n")

            for i in range(nlines):
                _att_str = "%d" % (i + 1)
                for key in cell_atts.keys():
                    _att_str += " " + str(cell_atts[key][i])
                _att_str += "\n"
                f.write(_att_str)

        f.write("\n")


class Infiles:
    def _surf_mesh_backup(in_name, out_name, skip_sort=False):
        # Driver for producing a surface mesh from
        # a prism mesh

        if skip_sort:
            infile = """read/avs/{0}/mo1
resetpts/itp
extract/surfmesh/1,0,0/mo2/mo1/external
dump/avs/{1}/mo2

finish""".format(
                in_name, out_name
            )
            return infile

        infile = """read/avs/{0}/mo1
resetpts/itp


createpts/median

sort/mo1/index/ascending/ikey/itetclr zmed ymed xmed

reorder/mo1/ikey
cmo/DELATT/mo1/ikey
cmo/DELATT/mo1/xmed
cmo/DELATT/mo1/ymed
cmo/DELATT/mo1/zmed
cmo/DELATT/mo1/ikey

extract/surfmesh/1,0,0/mo2/mo1/external
dump/avs/{1}/mo2

finish
""".format(
            in_name, out_name
        )
        return infile

    # user_function
    distance_field = """cmo/DELATT/mo_pts/dfield
compute / distance_field / mo_pts / mo_line_work / dfield
math/multiply/mo_pts/x_four/1,0,0/mo_pts/dfield/PARAM_A/
math/add/mo_pts/x_four/1,0,0/mo_pts/x_four/PARAM_B/
cmo/copyatt/mo_pts/mo_pts/fac_n/x_four
finish
"""
    # user_function2
    distance_field_2 = """cmo/DELATT/mo_pts/dfield
compute / distance_field / mo_pts / mo_line_work / dfield
math/multiply/mo_pts/x_four/1,0,0/mo_pts/dfield/PARAM_A2/
math/add/mo_pts/x_four/1,0,0/mo_pts/x_four/PARAM_B2/
cmo/copyatt/mo_pts/mo_pts/fac_n/x_four
finish
"""

    # infile_get_facesets3
    get_facesets3 = """# get default facesets bottom, top, sides

# FIX so MO has same numbering as exodus mesh
# use sort to order element blocks as exodus will order
# if this is not done, lagrit faceset numbers will not
# correlate to exodus faceset numbers
# itetclr must be ordered correctly

# sort based on element itetclr number and median location
# save median points to check they are inside mesh
cmo status CMO_PRISM brief
cmo select CMO_PRISM
createpts / median
sort / CMO_PRISM / index / ascending / ikey / itetclr xmed ymed zmed
reorder / CMO_PRISM / ikey
  cmo / DELATT / CMO_PRISM / ikey

# get outside surface mesh
extract/surfmesh/1,0,0/motmp_s/CMO_PRISM/external
cmo select motmp_s

#################################################
# BEGIN facesets based on layer and river surface

# Default value for all sides is 3
cmo /setatt/ motmp_s / itetclr 3

# bottom
cmo select motmp_s
pset/p1/attribute/layertyp/1,0,0/-1/eq
eltset/e1/exclusive/pset/get/p1
cmo/setatt/motmp_s/itetclr eltset,get,e1 1
cmo/copy/mo_tmp1/motmp_s
cmo/DELATT/mo_tmp1/itetclr0
cmo/DELATT/mo_tmp1/itetclr1
cmo/DELATT/mo_tmp1/facecol
cmo/DELATT/mo_tmp1/idface0
cmo/DELATT/mo_tmp1/idelem0
eltset/eall/itetclr/ge/0
eltset/edel/not eall e1
rmpoint/element/eltset get edel
rmpoint/compress
dump/avs2/fs1_bottom.avs/mo_tmp1/0 0 0 2

# top
cmo/delete/mo_tmp1
cmo select motmp_s
pset/p2/attribute/layertyp/1,0,0/-2/eq
eltset/e2/exclusive/pset/get/p2
cmo/setatt/motmp_s/itetclr eltset,get,e2 2
cmo/copy/mo_tmp1/motmp_s
cmo/DELATT/mo_tmp1/itetclr0
cmo/DELATT/mo_tmp1/itetclr1
cmo/DELATT/mo_tmp1/facecol
cmo/DELATT/mo_tmp1/idface0
cmo/DELATT/mo_tmp1/idelem0
eltset/eall/itetclr/ge/0
eltset/edel/not eall e2
rmpoint/element/eltset get edel
rmpoint/compress
dump/avs2/fs2_top.avs/mo_tmp1/0 0 0 2
dump gmv tmp_top.gmv mo_tmp1
cmo/delete/mo_tmp1

# sides - all sides, no direction
cmo select motmp_s
cmo/copy/mo_tmp1/motmp_s
cmo/DELATT/mo_tmp1/itetclr0
cmo/DELATT/mo_tmp1/itetclr1
cmo/DELATT/mo_tmp1/facecol
cmo/DELATT/mo_tmp1/idface0
cmo/DELATT/mo_tmp1/idelem0
eltset/edel/ itetclr lt 3
rmpoint/element/eltset get edel
rmpoint/compress
dump/avs2/fs3_sides_all.avs/mo_tmp1/0 0 0 2
dump gmv tmp_sides.gmv mo_tmp1
cmo/delete/mo_tmp1

###################################
# At this point mesh facesets are set for default
# bottom=1, top=2, sides=3
# fs1_bottom.avs fs2_top.avs fs3_sides_all.avs

finish
"""


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


def build_refined_triplane(
    dem_raster: Raster,
    refinement_feature: Shape,
    min_edge_length: float,
    max_edge_length: float,
    delta: float = 0.75,
    slope: float = 2.0,
    refine_dist: float = 0.5,
    verbose: bool = False,
    outfile: str = None,
):
    # boundary:np.ndarray,
    # feature:np.ndarray,
    # h:float,
    # connectivity:bool=None,
    # delta:float=0.75,
    # slope:float=2.,
    # refine_dist:float=0.5,
    # outfile:str=None):
    """
    Constructs a triplane mesh refined around a feature using LaGriT
    as a backend.

    Requires an Nx2 np.ndarray as a boundary input, and an Nx2 np.ndarray
    as a feature input.
    """

    from pylagrit import PyLaGriT

    lg = PyLaGriT(verbose=verbose)
    boundary_distance = max_edge_length
    a = min_edge_length * 1.75
    h = min_edge_length * 1.75

    boundary, connectivity = dem_raster.get_boundary(distance=boundary_distance)
    # feature = filter_points(deepcopy(refinement_feature.points), a)
    feature = deepcopy(refinement_feature.points)

    # Define initial parameters
    counterclockwise = False
    h_eps = h * 10 ** -7
    PARAM_A = slope
    PARAM_B = h * (1 - slope * refine_dist)
    PARAM_A2 = 0.5 * slope
    PARAM_B2 = h * (1 - 0.5 * slope * refine_dist)

    if connectivity is None:
        connectivity = line_connectivity(boundary)

    # cfg.log.debug('Writing boundary to poly_1.inp')
    write_line(boundary, "poly_1.inp", connections=connectivity)

    # cfg.log.debug('Writing feature to intersections_1.inp')
    write_line(feature, "intersections_1.inp")

    # Write massage macros
    with open("user_function.lgi", "w") as f:
        f.write(Infiles.distance_field)

    with open("user_function2.lgi", "w") as f:
        f.write(Infiles.distance_field_2)

    # cfg.log.info('Preparing feature and boundary')

    # Read boundary and feature
    mo_poly_work = lg.read("poly_1.inp", name="mo_poly_work")
    mo_line_work = lg.read("intersections_1.inp", name="mo_line_work")

    # Triangulate Fracture without point addition
    mo_pts = mo_poly_work.copypts(elem_type="triplane")
    mo_pts.select()

    # cfg.log.info('First pass triangulation')

    if counterclockwise:
        mo_pts.triangulate(order="counterclockwise")
    else:
        mo_pts.triangulate(order="clockwise")

    # Set element attributes for later use
    mo_pts.setatt("imt", 1, stride=(1, 0, 0))
    mo_pts.setatt("itetclr", 1, stride=(1, 0, 0))
    mo_pts.resetpts_itp()

    mo_pts.select()

    # Refine at increasingly smaller distances, approaching h
    for (i, ln) in enumerate([8, 16, 32, 64][::-1]):
        # cfg.log.info('Refining at length %s' % str(ln))

        h_scale = ln * h
        perturb = h_scale * 0.05

        mo_pts.massage(h_scale, h_eps, h_eps)

        # Do a bit of smoothing on the first pass
        if i == 0:
            for _ in range(3):
                mo_pts.recon(0)
                mo_pts.smooth()
            mo_pts.recon(0)

        mo_pts.resetpts_itp()

        # p_move = mo_pts.pset_attribute('itp',0,comparison='eq',stride=(1,0,0),name='p_move')
        # p_move.perturb(perturb,perturb.format(ln),0.0)

        # Smooth and reconnect
        for _ in range(6):
            mo_pts.recon(0)
            mo_pts.smooth()
        mo_pts.recon(0)

    # Define attributes to be used for massage functions
    mo_pts.addatt("x_four", vtype="vdouble", rank="scalar", length="nnodes")
    mo_pts.addatt("fac_n", vtype="vdouble", rank="scalar", length="nnodes")

    # Define internal variables for user_functions
    lg.define(
        mo_pts=mo_pts.name,
        PARAM_A=PARAM_A,
        PARAM_A2=PARAM_A2,
        PARAM_B=PARAM_B,
        PARAM_B2=PARAM_B2,
    )

    # cfg.log.info('Smoothing mesh (1/2)')

    # Run massage2
    mo_pts.dump("surface_lg.inp")
    mo_pts.massage2(
        "user_function2.lgi",
        0.8 * h,
        "fac_n",
        0.00001,
        0.00001,
        stride=(1, 0, 0),
        strictmergelength=True,
    )

    lg.sendline("assign///maxiter_sm/1")

    for _ in range(3):
        mo_pts.smooth()
        mo_pts.recon(0)

    # cfg.log.info('Smoothing mesh (2/2)')

    # Massage once more, cleanup, and return
    lg.sendline("assign///maxiter_sm/10")
    mo_pts.massage2(
        "user_function.lgi",
        0.8 * h,
        "fac_n",
        0.00001,
        0.00001,
        stride=(1, 0, 0),
        strictmergelength=True,
    )

    mo_pts.delatt("rf_field_name")

    mo_line_work.delete()
    mo_poly_work.delete()

    mo_pts.dump("surface_lg.inp")

    # TODO: change to Exodus!
    mo = read_avs(
        "surface_lg.inp",
        keep_material_id=False,
        keep_node_attributes=False,
        keep_cell_attributes=False,
    )

    cleanup(
        [
            "poly_1.inp",
            "surface_lg.inp",
            "lagrit.log",
            "lagrit.out",
            "user_function.lgi",
            "user_function2.lgi",
        ]
    )

    z_values = map_elevation(dem_raster, mo.nodes)
    mo.nodes[:, 2] = z_values

    return mo
