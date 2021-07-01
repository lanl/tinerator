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
from typing import List, Union, Any
from ..logging import log, warn, debug, error

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
    "Hex8",
    "Hex9",
    "Hex20",
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


def debug_side_sets(side_set, x, y, z, mesh_cells):
    name = side_set["set_name"]
    num_points = len(x)
    num_cells = len(side_set["set_elems"])

    set_elems = mesh_cells[side_set["set_elems"] - 1]

    with open(f"{name}.inp", "w") as f:
        f.write(f"{num_points} {num_cells} 0 0 0\n")
        for i in range(num_points):
            f.write(f"{i+1} {x[i]} {y[i]} {z[i]}\n")

        for i in range(num_cells):
            cell_type = "prism"
            cell_conn = " ".join([str(x) for x in set_elems[i]])
            f.write(f"{i+1} 1 {cell_type} {cell_conn}\n")


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


def _get_key(dictionary: dict, key: str, fill_none: Any = None):
    value = dictionary[key] if key in dictionary else None

    if (fill_none is not None) and (value is None):
        value = fill_none

    return value


def prepare_cell_blocks():
    raise NotImplementedError()


def prepare_node_sets(node_sets: List[dict]) -> List[dict]:
    node_sets_exo = []

    if isinstance(node_sets, dict):
        node_sets = [node_sets]
    elif node_sets is None:
        return []

    for (i, node_set) in enumerate(node_sets):
        set_id = _get_key(node_set, "set_id", fill_none=int(f"4{i}"))
        set_name = _get_key(node_set, "set_name", fill_none=f"NodeSetID={set_id}")
        set_nodes = _get_key(node_set, "set_nodes")
        num_set_nodes = len(set_nodes)
        num_set_dist_facts = _get_key(node_set, "num_set_dist_facts", fill_none=0)

        node_sets_exo.append(
            {
                "set_name": set_name,
                "set_id": set_id,
                "set_nodes": set_nodes,
                "num_set_nodes": num_set_nodes,
                "num_set_dist_facts": num_set_dist_facts,
            }
        )

    return node_sets_exo


def prepare_side_sets(side_sets, cell_mapping):
    """
    Prepares side sets to local Exodus
    reference frame.

    Args:
        side_sets (List[dict]): List of dictionaries.
        cell_mapping ([type]): n]

    Returns:
        [type]: [description]
    """
    side_sets_exo = []

    if isinstance(side_sets, dict):
        side_sets = [side_sets]
    elif side_sets is None:
        return []

    set_mapping = sorted(list(range(len(cell_mapping))), key=lambda x: cell_mapping[x])
    set_mapping = np.array(set_mapping)

    for (i, side_set) in enumerate(side_sets):
        set_id = _get_key(side_set, "set_id", fill_none=int(f"3{i}"))
        set_name = _get_key(side_set, "set_name", fill_none=f"SideSetID={set_id}")
        set_elems = np.array(_get_key(side_set, "set_elems"), dtype=int)
        set_sides = np.array(_get_key(side_set, "set_sides"), dtype=int)
        num_set_sides = len(set_sides)
        num_set_dist_facts = _get_key(side_sets, "num_set_dist_facts", fill_none=0)

        print(f"{set_name}: min = {np.min(set_elems)}; max = {np.max(set_elems)}")

        side_sets_exo.append(
            {
                "set_name": set_name,
                "set_id": set_id,
                "set_elems": set_mapping[set_elems - 1] + 1,
                "set_sides": set_sides.astype(int),
                "num_set_sides": num_set_sides,
                "num_set_dist_facts": num_set_dist_facts,
            }
        )

    return side_sets_exo


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
    write_set_names: bool = True,
    element_block_name_prefix: str = "MATERIAL_ID_",
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
        warn("Cell block IDs were not passed: mesh will be one block")
        cell_block_ids = np.full((num_elems,), 1, dtype=INT_TYPE)

    assert cell_block_ids.shape[0] == num_elems, "Each cell must have a block ID"

    if mesh_elems.shape[1] == 6:
        elem_type = "Wedge6"
    elif mesh_elems.shape[1] == 3:
        elem_type = "Tri3"
    elif mesh_elems.shape[1] == 4:
        elem_type = "Quad4"
    elif mesh_elems.shape[1] == 8:
        elem_type = "Hex8"
    else:
        raise ValueError("Unknown element shape")

    assert elem_type.upper() in [x.upper() for x in EXODUS_ELEMENTS]

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
        # TODO: move to prepare cell blocks
        mask = cell_block_ids == block_id

        connectivity = mesh_elems[mask].astype(INT_TYPE)
        connectivity = connectivity[:, element_mapping[elem_type.upper()]]

        blocks.append(
            {
                "connectivity": connectivity.flatten(order="C"),
                "block_id": block_id,
                "block_name": f"{element_block_name_prefix}{block_id}",
                "elem_type": elem_type.upper(),
                "num_elem_this_blk": connectivity.shape[0],
                "num_nodes_per_elem": connectivity.shape[1],
                "num_attr": 0,
            }
        )

    # ======================
    # Configuring sets
    # ======================
    side_sets = prepare_side_sets(side_sets, cell_mapping=mapping)
    node_sets = prepare_node_sets(node_sets)
    element_sets = []

    num_side_sets = len(side_sets)
    num_node_sets = len(node_sets)
    num_elem_sets = len(element_sets)

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

    for block in blocks:
        exo_id.put_elem_connectivity(block["block_id"], block["connectivity"])

    # -------------------------------------------------
    # SECTION WRITE Node Sets
    if num_node_sets > 0:
        if write_set_names:
            exo_id.put_node_set_names([ns["set_name"] for ns in node_sets])

        for ns in node_sets:
            exo_id.put_node_set_params(
                ns["set_id"], ns["num_set_nodes"], ns["num_set_dist_facts"]
            )
            exo_id.put_node_set(ns["set_id"], ns["set_nodes"])

    # -------------------------------------------------
    # SECTION WRITE Elem Sets
    if num_elem_sets > 0:
        raise NotImplementedError()

    # -------------------------------------------------
    # SECTION WRITE Side Sets
    if num_side_sets > 0:
        if write_set_names:
            exo_id.put_side_set_names([ss["set_name"] for ss in side_sets])

        for ss in side_sets:
            debug(
                f"{ss['set_name']}: min_elem = {np.min(ss['set_elems'])}; max_elem = {np.max(ss['set_elems'])}"
            )
            exo_id.put_side_set_params(
                ss["set_id"], ss["num_set_sides"], ss["num_set_dist_facts"]
            )
            exo_id.put_side_set(ss["set_id"], ss["set_elems"], ss["set_sides"])

    # -------------------------------------------------
    # SECTION DONE with ExodusII file

    exo_id.close()

    print("EXODUS write was successful.")
