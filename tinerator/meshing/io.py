import numpy as np

def write_exodus(
    outfile: str,
):
    """
    Write a mesh to an Exodus file.
    """
    
    # Option 1: https://salvushub.github.io/pyexodus/
    # Option 2: https://github.com/nschloe/meshio

    pass

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
    with open(outfile, "w") as f:

        n_nodes = nodes.shape[0]
        n_cells = cells.shape[0] if cells is not None else 0
        n_node_attrbs = (
            len(node_attributes.keys()) if node_attributes is not None else 0
        )
        n_cell_attrbs = (
            len(cell_attributes.keys()) if cell_attributes is not None else 0
        )

        # Write out the header
        # nodes, cells, node attributes, cell attributes, 0
        f.write(
            "{} {} {} {} 0\n".format(
                n_nodes, n_cells, n_node_attrbs, n_cell_attrbs
            )
        )

        # Write out the nodes
        # index, x, y, z
        for (i, node) in enumerate(nodes):
            f.write("{} {} {} {}\n".format(i + 1, *nodes[i]))

        if matid is None and len(cells) > 0:
            matid = [1] * len(cells)

        # Write out the cells
        # index, material id, cell type, ...connectivity
        for (i, cell) in enumerate(cells):
            f.write(
                "{} {} {} {}\n".format(
                    i + 1,
                    int(matid[i]),
                    cname,
                    " ".join([str(x) for x in list(map(int, cells[i]))]),
                )
            )

        # Write out node attributes
        if n_node_attrbs > 0:
            # number of node attributes, 1 ... 1
            f.write(
                "{} {}\n".format(
                    n_node_attrbs, " ".join(["1"] * n_node_attrbs)
                )
            )

            # attribute name, data type <integer, real>
            for key in node_attributes.keys():
                attrb_type = "integer"
                f.write("{}, {}\n".format(key, attrb_type))

            # node index, all attribute values at that node
            for i in range(n_nodes):
                attribute_row = []

                for key in node_attributes.keys():
                    attribute_row.append(str(node_attributes[key]["data"][i]))

                f.write("%d %s\n" % (i + 1, " ".join(attribute_row)))
        
        if n_cell_attrbs > 0:
            print("Cell attributes aren\'t supported right now.")