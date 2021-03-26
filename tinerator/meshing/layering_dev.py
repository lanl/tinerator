import os
import numpy as np
import tempfile
import meshio
from pylagrit import PyLaGriT
from copy import copy, deepcopy
from enum import Enum, auto
from .mesh import Mesh, StackedMesh, ElementType, load_mesh
from ..logging import log, warn, debug, _pylagrit_verbosity


def extract_surface_mesh(mesh):  #: Mesh):
    """
    Extracts the boundary of a mesh. For a solid (volume) mesh,
    it extracts the surface mesh. If it is a surface mesh, it
    extracts the edge mesh.

    TODO: the current issue is that a surf mesh for a prism is
    multi-material (tri for surface; quad for sides.)
    """

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = "/Users/livingston/playground/tinerator/tmp/newtmp"
        mesh.save(os.path.join(tmp_dir, "volmesh.inp"))

        debug("Launching PyLaGriT")
        lg = PyLaGriT(verbose=_pylagrit_verbosity(), cwd=tmp_dir)
        lg.sendline("read/avs/volmesh.inp/mo1")

        # if SORT: ......

        lg.sendline("extract/surfmesh/1,0,0/mo2/mo1/external")
        lg.sendline("dump/avs/surfmesh_lg.inp/mo2")
        del lg

        # surf_mesh = load_mesh(os.path.join(tmp_dir, 'surfmesh_lg.inp'))
        surf_mesh = meshio.read(
            os.path.join(tmp_dir, "surfmesh_lg.inp"), file_format="avsucd"
        )

    return surf_mesh


def DEV_basic_facesets(prism_mesh: Mesh):

    # compass_map = {
    #    "bottom": 1,
    #    "top": 2,
    # }

    # LAYERTYP_MAP = {
    #    "top": -2,
    #    "bottom": -1,
    #    "sides": 0,
    # }

    surf_mo = extract_surface_mesh(prism_mesh)
    layertyp = surf_mo.point_data["layertyp"]

    layertyp_cells = np.zeros((len(surf_mo.cells),), dtype=int)

    aaa = []
    for i in range(len(surf_mo.cells)):
        aaa.extend(np.mean(layertyp[surf_mo.cells[i].data], axis=1))

    bbb = []
    for i in range(len(aaa)):
        if np.allclose(aaa[i], -2):
            bbb.append(-2)
        elif np.allclose(aaa[i], -1):
            bbb.append(-1)
        else:
            bbb.append(0)

    layertyp_cells = np.array(bbb, dtype=int)
    id_elem = np.hstack(surf_mo.cell_data["idelem1"])
    id_face = np.hstack(surf_mo.cell_data["idface1"])

    top_cells = id_elem[layertyp_cells == -2]
    bottom_cells = id_elem[layertyp_cells == -1]
    side_cells = id_elem[layertyp_cells == 0]

    print(id_face, top_cells, bottom_cells, side_cells)

    # top_cells_idx = np.argwhere(layertyp_cells == -2).T
    # top_cells_id = np.hstack(id_elem[top_cells_idx][0])
    # top_cells_faces = np.hstack(id_face[top_cells_idx][0])

    # bottom_cells_idx = np.argwhere(layertyp_cells == -1).T
    # bottom_cells_id = np.hstack(id_elem[bottom_cells_idx][0])
    # bottom_cells_faces = np.hstack(id_face[bottom_cells_idx][0])

    # side_cells_idx = np.argwhere(layertyp_cells == 0).T
    # side_cells_id = np.hstack(id_elem[side_cells_idx][0])
    # side_cells_faces = np.hstack(id_face[side_cells_idx][0])

    assert 1 == 0

    import ipdb

    ipdb.set_trace()

    return surf_mo


def DEV_get_dummy_layers(surface_mesh, depths):
    layers = []

    if depths[0] != 0.0:
        depths = [0.0] + list(depths)

    current_depth = 0.0

    for depth in depths:
        connectivity = deepcopy(surface_mesh.elements)
        nodes = deepcopy(surface_mesh.nodes)
        current_depth += depth
        nodes[:, 2] -= current_depth

        m = Mesh(
            etype=surface_mesh.element_type,
            crs=surface_mesh.crs,
            nodes=nodes,
            elements=connectivity,
        )

        layers.append(m)

    return layers


def DEV_stack(layers: list, matids: list = None):

    # Create and chdir to a temporary dir,
    # so that LaGriT artifacts aren't saved
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = "/Users/livingston/playground/tinerator/tmp/example/ttt/asdfasd"
        debug("Launching PyLaGriT")
        lg = PyLaGriT(verbose=_pylagrit_verbosity(), cwd=tmp_dir)

        if matids is None:
            matids = list(range(1, len(layers)))

        assert len(matids) == len(layers) - 1

        layer_files = []

        for (i, layer) in enumerate(layers[::-1]):
            layer_out = "layer%d.inp" % i

            layer.save(os.path.join(tmp_dir, layer_out))
            layer_files.append(layer_out)

        log("Adding volume to layers")
        debug(f"Layer files: {str(layer_files)}")

        stack = lg.create()
        stack.stack_layers(
            layer_files,
            flip_opt=True,
            nlayers=[""] * len(layer_files),
            matids=matids[::-1] + [-1],
        )

        debug("Filling layers with `stack_fill`")
        cmo_prism = stack.stack_fill()
        cmo_prism.resetpts_itp()

        debug("Writing volume mesh to disk: volume.inp")
        cmo_prism.dump("volume.inp")

        del lg  # close the LaGriT process

        volume_mesh = load_mesh(os.path.join(tmp_dir, "volume.inp"))

    return volume_mesh
