# https://gsjaardema.github.io/seacas-docs/exodusII-new.pdf
# /Users/livingston/playground/lanl/tinerator/tpl/seacas/install/lib:
# https://github.com/ecoon/watershed-workflow/blob/c1b593e79e96b8fe22685d38c8777df0d824b1f6/workflow/mesh.py

"""
References:

[0]: ExodusII API: https://gsjaardema.github.io/seacas-docs/exodusII-new.pdf
[1]: LaGriT `dumpexodusII.f`: https://github.com/lanl/LaGriT/blob/master/src/dumpexodusII.f#L1351
"""

import os
import numpy as np

# See the ExodusII API [0] for node ordering
EXODUS_ELEMENTS = [
    "Bar2",
    "Bar3",  # pp. 15
    "Tri3",
    "Tri4",
    "Tri6",
    "Tri7",  # pp. 16
    "Quad4",
    "Quad5",
    "Quad8",
    "Quad9",  # pp. 17
    "Tet4",
    "Tet5",
    "Tet10",
    "Tet11",
    "Tet14",
    "Tet15",  # pp. 18-20
    "Pyra5",
    "Pyra13",
    "Pyra14",  # pp. 20
    "Wedge6",
    "Wedge15",
    "Wedge16",  # pp. 21
]

# Mapping between local element nums from LaGriT to Exodus II

#  integer lag2exo_nmap(8,8)
#  data lag2exo_nmap / 1, 0, 0, 0, 0, 0, 0, 0, ! Point
# &                    1, 2, 0, 0, 0, 0, 0, 0, ! Line
# &                    1, 2, 3, 0, 0, 0, 0, 0, ! Tri
# &                    1, 2, 3, 4, 0, 0, 0, 0, ! Quad
# &                    1, 2, 3, 4, 0, 0, 0, 0, ! Tet
# &                    0, 0, 0, 0, 0, 0, 0, 0, ! Pyramid
# &                    1, 2, 3, 4, 5, 6, 0, 0, ! Wedge/Prism
# &                    1, 2, 3, 4, 5, 6, 7, 8/ ! Hex

EXODUS_ELEMENT_MAPPING = {"TRI3": None, "WEDGE6": [3, 4, 5, 0, 1, 2]}

# Mapping between local face nums from LaGriT to Exodus II

#  integer lag2exo_fmap(6,8)
#  data lag2exo_fmap / 1, 0, 0, 0, 0, 0, ! Point
# &                    1, 2, 0, 0, 0, 0, ! Line
# &                    2, 3, 1, 0, 0, 0, ! Tri
# &                    1, 2, 3, 4, 0, 0, ! Quad
# &                    2, 3, 1, 4, 0, 0, ! Tet
# &                    0, 0, 0, 0, 0, 0, ! Pyramid
# &                    4, 5, 1, 2, 3, 0, ! Wedge/Prism
# &                    5, 6, 1, 2, 3, 4/ ! Hex

EXODUS_FACE_MAPPING = {"WEDGE6": [4, 5, 1, 2, 3]}


def check_mesh_diff(mesh1_filename: str, mesh2_filename: str, print_diff: bool = True):
    """
    WARNING: this function uses `os.system` and could be used for
    hostile purposes. Make *sure* your filenames are valid paths.

    Some filename checking is done to try to prevent this.

    Example:
    >> mesh1_filename="somemesh.exo; sudo rm -rf /"
    >> os.system(f"ncdump {mesh1_filename} >> {out1}")

    This maps to:
    >> os.system("ncdump somemesh.exo; sudo rm -rf / >> mesh1.txt")

    Or, equivalently:
    >> os.system("ncdump somemesh.exo")
    >> os.system("sudo rm -rf / >> mesh1.txt")
    """
    import tempfile

    assert os.path.exists(mesh1_filename)
    assert os.path.exists(mesh2_filename)

    with tempfile.TemporaryDirectory() as tmp_dir:
        out1 = os.path.join(tmp_dir, "mesh1.txt")
        out2 = os.path.join(tmp_dir, "mesh2.txt")

        os.system(f"ncdump {mesh1_filename} >> {out1}")
        os.system(f"ncdump {mesh2_filename} >> {out2}")
        diff = os.popen(f"diff {out1} {out2}").read()

    if print_diff:
        print(diff)

    return diff.replace("\t", "").split("\n")


def exodus_cell_remapping(
    cells: np.ndarray, block_id: np.ndarray, nodes: np.ndarray
) -> np.ndarray:
    """
    The Exodus API requires mesh cells to be sorted
    by, in successive order:
    - Block ID (material ID)
    - Cell centroid -> Z coord
    - Cell centroid -> Y coord
    - Cell centroid -> X coord

    This returns a mapping for cells into this format.
    """
    import operator

    mapping = np.array(list(range(cells.shape[0])))
    cell_centroids = np.mean(nodes[cells], axis=1)

    # Sort by: matID, z_med, y_med, x_med
    key = np.hstack(
        [mapping[:, None], block_id[:, None], np.fliplr(cell_centroids)]
    ).tolist()
    key.sort(key=operator.itemgetter(1, 2, 3, 4))

    return np.array(key)[:, 0].astype(int)


def dump_exodus(
    outfile: str,
    mesh_nodes: np.ndarray,
    mesh_cells: np.ndarray,
    cell_block_ids: np.ndarray = None,
    side_sets: list = None,
    node_sets: list = None,
    element_sets: list = None,
    mesh_title: str = "TINerator Mesh",
    clobber_existing_file: bool = True,
    element_mapping: dict = EXODUS_ELEMENT_MAPPING,
):
    """
    Writes nodes and elements to an Exodus-format mesh.

    mesh_nodes[num_nodes, 3]: The mesh nodes array
    mesh_cells[num_cells, N]: The mesh cells array
    cell_block_ids[num_cells]: A vector of length `num_cells`
    """

    import exodus3 as exodus

    INT_TYPE = int
    FLOAT_TYPE = float

    # EXODUS will throw an error trying to overwrite an existing file:
    # "Exception: ERROR: Cowardly not opening {outfile} for write. File already exists."
    # Remove the file manually.
    if clobber_existing_file and os.path.exists(outfile):
        os.remove(outfile)

    num_dim = 3
    array_type = "numpy"

    mesh_nodes = np.array(mesh_nodes).astype(FLOAT_TYPE)
    mesh_elems = np.array(mesh_cells).astype(INT_TYPE)

    assert mesh_nodes.shape[1] == num_dim, "Mesh nodes must have X, Y, Z columns"
    assert np.min(mesh_elems) != 0, "Connectivity is 1-indexed, not 0-indexed"

    num_nodes = mesh_nodes.shape[0]
    num_elems = mesh_elems.shape[0]

    if cell_block_ids is None:
        cell_block_ids = np.full((num_elems,), 1, dtype=INT_TYPE)

    assert cell_block_ids.shape[0] == num_elems, "Each cell must have a block ID"

    if mesh_elems.shape[1] == 6:
        elem_type = "Wedge6"
    elif mesh_elems.shape[1] == 3:
        elem_type = "Tri3"
    else:
        raise ValueError("Unknown element shape")

    assert elem_type.upper() in [x.upper() for x in EXODUS_ELEMENTS]

    num_node_sets = 0
    num_elem_sets = 0

    # Gather coordinate data
    coord_names = ["coordX", "coordY", "coordZ"]
    x_coor = mesh_nodes[:, 0].astype(FLOAT_TYPE)
    y_coor = mesh_nodes[:, 1].astype(FLOAT_TYPE)
    z_coor = mesh_nodes[:, 2].astype(FLOAT_TYPE)

    mapping = exodus_cell_remapping(mesh_elems - 1, cell_block_ids, mesh_nodes)

    mesh_elems = mesh_elems[mapping]
    cell_block_ids = cell_block_ids[mapping]
    block_ids = cell_block_ids[np.unique(cell_block_ids, return_index=True)[1]]

    num_elem_blk = len(block_ids)

    # Here, we're going to create blocks into a psuedo-data structure
    # using dictionaries. This will make very simple to pass the data
    # to the Exodus API.
    blocks = []

    for block_id in block_ids:
        mask = cell_block_ids == block_id

        connectivity = mesh_elems[mask].astype(INT_TYPE)
        connectivity = connectivity[:, element_mapping[elem_type.upper()]]

        blocks.append(
            {
                "connectivity": connectivity.flatten(order="C"),
                "block_id": block_id,
                "block_name": f"blockID={block_id}",
                "elem_type": elem_type.upper(),
                "num_elem_this_blk": connectivity.shape[0],
                "num_nodes_per_elem": connectivity.shape[1],
                "num_attr": 0,
            }
        )

    # ======================
    # Configuring side sets
    # ======================
    num_side_sets = 0 if side_sets is None else len(side_sets)

    side_sets_exo = []
    if num_side_sets > 0:
        for (i, ss) in enumerate(side_sets):
            name = ss.name
            setid = ss.setid

            if setid is None:
                setid = int(f"3{i}")

            if name is None or name.strip() == "":
                name = f"SideSetID={setid}"

            face_map = np.array(EXODUS_FACE_MAPPING[elem_type.upper()])

            ss_elems = np.sort(mapping[ss.elem_list - 1] + 1)
            ss_sides = face_map[ss.side_list - 1]

            assert len(ss_elems) == len(ss_sides)

            side_sets_exo.append(
                {
                    "name": name,
                    "side_set_id": setid,
                    "num_ss_sides": len(ss_elems),
                    "num_ss_dist_facts": 0,
                    "ss_elems": ss_elems,
                    "ss_sides": ss_sides,
                }
            )

    # -------------------------------------------------
    # SECTION BEGIN Exodus mesh write

    title = mesh_title.encode("ascii")
    ex_pars = exodus.ex_init_params(
        num_dim=num_dim,
        num_nodes=num_nodes,
        num_elem=num_elems,
        num_elem_blk=num_elem_blk,
        num_node_sets=num_node_sets,
        num_side_sets=num_side_sets,
    )

    exo_id = exodus.exodus(
        outfile, mode="w", array_type=array_type, title=title, init_params=ex_pars
    )

    # -------------------------------------------------
    # SECTION WRITE QA Information
    # Put some QA info - problem name, date, time etc.

    cmo_name = "CMO_NAME"
    records = [(cmo_name, "probname", "Today", "Time")]
    exo_id.put_qa_records(records)

    # -------------------------------------------------
    # SECTION WRITE Coordinates

    exo_id.put_coord_names(coord_names)
    exo_id.put_coords(x_coor, y_coor, z_coor)

    # -------------------------------------------------
    # SECTION WRITE Element Blocks
    # The function ex_put_elem_block (or EXPELB for Fortran) writes the
    # parameters used to describe an element block.

    for block in blocks:
        exo_id.put_elem_blk_info(
            block["block_id"],
            block["elem_type"],
            block["num_elem_this_blk"],
            block["num_nodes_per_elem"],
            block["num_attr"],
        )

    block_names = [block["block_name"] for block in blocks]
    exo_id.put_elem_blk_names(block_names)

    # -------------------------------------------------
    # SECTION WRITE Element Connectivity

    for block in blocks:
        exo_id.put_elem_connectivity(block["block_id"], block["connectivity"])

    if num_node_sets > 0:
        # -------------------------------------------------
        # SECTION WRITE node sets (pset) (if defined)

        # int ex_put_node_set_param(int exoid, int node_set_id, int num_nodes_in_set, int num_dist_in_set)
        #   exoid: EXODUS file ID
        #   node_set_id: The node set ID
        #   num_nodes_in_set: The number of nodes in the node set
        #   num_dist_in_set: The number of distribution factors in the node set.
        #       This should be either 0 for no factors, or should equal num_nodes_in_set.

        # EXPNP(idexo, nodeset_tag(i), nnodes_ns(i), 0, status)
        # status = ex_put_node_set_param(exoid, node_set_id, num_nodes_in_set, num_dist_in_set)

        # int ex_put_node_set(int exoid, int node_set_id, int *node_set_node_list)
        #   exoid: EXODUS file ID
        #   node_set_id: The node set ID
        #   node_set_node_list: Array containing the node list for the node set.

        # EXPNS(idexo, nodeset_tag(i), outernodes(ibeg), status)
        # status = ex_put_node_set(exoid, node_set_id, node_set_node_list)

        # TODO: exo_put_sets should go here. Where is it?
        # exo_put_sets(
        #     idexo, EXNSET, trim(cpt3), ilen, set_id,
        #     mpno, 0, pmpary1, status
        # )
        print("Node sets: not implemented")

    if num_elem_sets > 0:
        # -------------------------------------------------
        # SECTION WRITE element sets (eltset) (if defined)
        # exo_put_sets(
        #     idexo, EX_ELEM_SET, trim(cpt3),
        #     icharlnf(trim(cpt3)), set_id, mpno, 0,
        #     exo_elt_ary, status
        # )
        print("Elem sets: not implemented")

    if num_side_sets > 0:
        # -------------------------------------------------
        # SECTION WRITE Side Sets (if defined)

        exo_id.put_side_set_names([ss["name"] for ss in side_sets_exo])

        for ss in side_sets_exo:
            exo_id.put_side_set_params(
                ss["side_set_id"], ss["num_ss_sides"], ss["num_ss_dist_facts"]
            )
            exo_id.put_side_set(ss["side_set_id"], ss["ss_elems"], ss["ss_sides"])

    # -------------------------------------------------
    # SECTION DONE with ExodusII file

    exo_id.close()

    print("EXODUS write was successful.")