import numpy as np
from ..logging import debug, log
from .meshing_types import AVS_TYPE_MAPPING, ElementType


def read_mpas(filename: str, load_dual_mesh: bool = True):
    """
    Reads an MPAS mesh.
    """
    from netCDF4 import Dataset
    
    with Dataset(filename, "r") as nc:

        try:
            on_a_sphere = True if nc.on_a_sphere.strip().lower() == "yes" else False
        except Exception:
            on_a_sphere = None

        debug(f"On a sphere? {on_a_sphere}")

        # Get some dimensions
        nCells = nc.dimensions["nCells"].size
        # nEdges = nc.dimensions["nEdges"].size
        nVertices = nc.dimensions["nVertices"].size

        if load_dual_mesh:
            vertices = np.zeros((nCells, 3), dtype=float)
            vertices[:, 0] = nc.variables["xCell"][:].data
            vertices[:, 1] = nc.variables["yCell"][:].data
            vertices[:, 2] = nc.variables["zCell"][:].data
            connectivity = nc.variables["cellsOnVertex"][:].data
        else:
            vertices = np.zeros((nVertices, 3), dtype=float)
            vertices[:, 0] = nc.variables["xVertex"][:].data
            vertices[:, 1] = nc.variables["yVertex"][:].data
            vertices[:, 2] = nc.variables["zVertex"][:].data
            connectivity = nc.variables["verticesOnCell"][:].data

    return vertices, connectivity


def read_avs(
    inp_filename: str,
):
    """
    Reads an AVS-UCD mesh file (extension: `.inp`) and returns a Mesh object.
    """

    nodes = elements = element_type = node_atts = elem_atts = None

    dtype_map = {"integer": int, "real": float}

    with open(inp_filename, "r") as f:
        header = map(int, f.readline().strip().split())
        n_nodes, n_elems, n_node_atts, n_elem_atts, _ = header

        if n_nodes:
            lines = []
            for _ in range(n_nodes):
                lines.append(f.readline().strip().split())
            lines = np.array(lines)

            nodes = lines[:, 1:].astype(float)

        if n_elems:
            lines = []
            for _ in range(n_elems):
                lines.append(f.readline().strip().split())
            lines = np.array(lines)

            material_id = lines[:, 1].astype(int)
            elements = lines[:, 3:].astype(int)
            element_type = AVS_TYPE_MAPPING[lines[0, 2].lower()]

        if n_node_atts:
            _ = f.readline()  # 00005  1  1  1  1  1

            names = [
                [x.strip() for x in f.readline().split(",")] for _ in range(n_node_atts)
            ]
            data = np.array(
                [f.readline().split()[1:] for _ in range(n_nodes)], dtype=float
            )
            node_atts = {
                names[i][0]: data[:, i].astype(dtype_map[names[i][1]])
                for i in range(n_node_atts)
            }

        if n_elem_atts:
            _ = f.readline()  # 00005  1  1  1  1  1

            names = [
                [x.strip() for x in f.readline().split(",")] for _ in range(n_elem_atts)
            ]
            data = np.array(
                [f.readline().split()[1:] for _ in range(n_elems)], dtype=float
            )
            elem_atts = {
                names[i][0]: data[:, i].astype(dtype_map[names[i][1]])
                for i in range(n_node_atts)
            }

    return nodes, elements, element_type, material_id, node_atts, elem_atts


def write_avs(
    outfile: str,
    nodes: np.ndarray,
    cells: np.ndarray,
    cname: str = "tri",
    matid: np.ndarray = None,
    node_attributes: dict = None,
    cell_attributes: dict = None,
):
    """
    Write a mesh to an AVS-UCD file.
    """

    write_list = lambda f, lst: f.write(" ".join(map(str, lst)) + "\n")

    with open(outfile, "w") as f:

        n_nodes = nodes.shape[0]
        n_cells = cells.shape[0] if cells is not None else 0
        n_node_atts = len(node_attributes) if node_attributes is not None else 0
        n_cell_atts = len(cell_attributes) if cell_attributes is not None else 0

        # Write out the header
        write_list(f, [n_nodes, n_cells, n_node_atts, n_cell_atts, 0])

        # Write out the nodes
        #    index, x, y, z
        if n_nodes:
            for (i, node) in enumerate(nodes):
                write_list(f, [i + 1, *node])

        # Write out the cells
        # index, material id, cell type, ...connectivity
        if n_cells:
            if matid is None:
                matid = [1] * n_cells

            for (i, cell) in enumerate(cells):
                write_list(f, [i + 1, matid[i], cname, *cell])

        if n_node_atts:
            atts = node_attributes
            att_names = node_attributes.keys()
            write_list(f, [n_node_atts] + [1] * n_node_atts)

            for name in att_names:
                if np.issubdtype(atts[name].dtype, np.integer):
                    f.write(f"{name}, integer\n")
                elif np.issubdtype(atts[name].dtype, np.inexact):
                    f.write(f"{name}, real\n")
                else:
                    raise ValueError("Unknown type")

            # node index, all attribute values at that node
            for i in range(n_nodes):
                write_list(f, [i + 1] + [atts[x][i] for x in att_names])

        if n_cell_atts:
            atts = cell_attributes
            att_names = cell_attributes.keys()
            write_list(f, [n_cell_atts] + [1] * n_cell_atts)

            for name in att_names:
                if np.issubdtype(atts[name].dtype, np.integer):
                    f.write(f"{name}, integer\n")
                elif np.issubdtype(atts[name].dtype, np.inexact):
                    f.write(f"{name}, real\n")
                else:
                    raise ValueError("Unknown type")

            # node index, all attribute values at that node
            for i in range(n_cells):
                write_list(f, [i + 1] + [atts[x][i] for x in att_names])
