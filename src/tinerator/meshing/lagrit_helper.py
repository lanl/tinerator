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
