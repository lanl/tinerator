import os
import numpy as np
import tempfile
import meshio
from pylagrit import PyLaGriT
from copy import copy, deepcopy
from enum import Enum, auto
from .mesh import Mesh, StackedMesh, ElementType, load_mesh
from ..logging import log, warn, debug, _pylagrit_verbosity


def DEV__driver_naive(lg, surface_mesh, top, bottom, sides):
    """
    Simple driver that constructs one or more of:
    * Top faceset
    * Bottom faceset
    * Side faceset
    """

    if isinstance(sides, bool):
        sides_simple = True
    elif isinstance(sides, dict):
        sides_simple = False
    else:
        raise ValueError("Unknown value for `sides`")

    faceset_fnames = []

    mo_surf = surface_mesh.copy()

    for att in ["itetclr0", "itetclr1", "facecol", "idface0", "idelem0"]:
        mo_surf.delatt(att)

    mo_surf.select()

    if not sides_simple:
        lg.sendline("settets/normal")
        compass_sets = {
            "north": mo_surf.eltset_attribute("itetclr", 4, boolstr="eq"),
            "south": mo_surf.eltset_attribute("itetclr", 6, boolstr="eq"),
            "east": mo_surf.eltset_attribute("itetclr", 5, boolstr="eq"),
            "west": mo_surf.eltset_attribute("itetclr", 3, boolstr="eq"),
        }

    mo_surf.setatt("itetclr", 3)

    ptop = mo_surf.pset_attribute("layertyp", -2, comparison="eq", stride=[1, 0, 0])
    pbot = mo_surf.pset_attribute("layertyp", -1, comparison="eq", stride=[1, 0, 0])

    etop = ptop.eltset(membership="exclusive")
    ebot = pbot.eltset(membership="exclusive")

    mo_surf.setatt("itetclr", 100, stride=["eltset", "get", etop.name])
    mo_surf.setatt("itetclr", 200, stride=["eltset", "get", ebot.name])

    esides = mo_surf.eltset_attribute("itetclr", 50, boolstr="lt")

    if top:
        debug("Generating top faceset")
        mo_tmp = mo_surf.copy()
        edel = mo_tmp.eltset_not([etop])
        mo_tmp.rmpoint_eltset(edel, resetpts_itp=False)

        fname = "fs_naive_top.avs"
        lg.sendline("dump / avs2 / " + fname + "/" + mo_tmp.name + "/ 0 0 0 2")
        mo_tmp.dump("DEBUG_naive_top_fs.inp")

        faceset_fnames.append(fname)
        mo_tmp.delete()

    if bottom:
        debug("Generating bottom faceset")
        mo_tmp = mo_surf.copy()
        edel = mo_tmp.eltset_not([ebot])
        mo_tmp.rmpoint_eltset(edel, resetpts_itp=False)

        fname = "fs_naive_bottom.avs"

        lg.sendline("dump / avs2 / " + fname + "/" + mo_tmp.name + "/ 0 0 0 2")

        mo_tmp.dump("DEBUG_naive_bottom_fs.inp")

        faceset_fnames.append(fname)
        mo_tmp.delete()

    if sides:
        if sides_simple:
            debug("Generating sides faceset")
            mo_tmp = mo_surf.copy()
            edel = mo_tmp.eltset_not([esides])
            mo_tmp.rmpoint_eltset(edel, resetpts_itp=False)

            fname = "fs_naive_sides.avs"

            lg.sendline("dump / avs2 / " + fname + "/" + mo_tmp.name + "/ 0 0 0 2")

            mo_tmp.dump("DEBUG_naive_sides_fs.inp")

            faceset_fnames.append(fname)
            mo_tmp.delete()

        else:

            for direction in ["north", "south", "west", "east"]:
                if not sides[direction]:
                    continue

                debug("Generating sides faceset: %s" % direction)

                mo_tmp = mo_surf.copy()
                edel = mo_tmp.eltset_not([compass_sets[direction]])
                mo_tmp.rmpoint_eltset(edel, resetpts_itp=False)

                fname = "fs_sides_%s.avs" % direction

                lg.sendline("dump / avs2 / " + fname + "/" + mo_tmp.name + "/ 0 0 0 2")

                mo_tmp.dump("DEBUG_sides_%s.inp" % direction)

                faceset_fnames.append(fname)
                mo_tmp.delete()

    mo_surf.delete()
    return faceset_fnames


def DEV_spit_out_simple_mesh(mesh):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = "/Users/livingston/playground/lanl/tinerator/tmp/exo"

        mesh.save(os.path.join(tmp_dir, "volmesh.inp"))

        mesh.save(os.path.join(tmp_dir, "BRONZE.inp"))

        debug("Launching PyLaGriT")

        print("1>>> ?")
        lg = PyLaGriT(verbose=_pylagrit_verbosity(), cwd=tmp_dir)
        print("2>>> ?")
        # lg.sendline("read/avs/volmesh.inp/mo1")
        stacked_mesh = lg.read("volmesh.inp")

        # ----------------- #

        stacked_mesh.resetpts_itp()
        lg.sendline("resetpts/itp")
        lg.sendline("createpts/median")
        lg.sendline(
            "sort/{0}/index/ascending/ikey/itetclr zmed ymed xmed".format(
                stacked_mesh.name
            )
        )
        lg.sendline("reorder/{0}/ikey".format(stacked_mesh.name))
        lg.sendline("cmo/DELATT/{0}/ikey".format(stacked_mesh.name))
        lg.sendline("cmo/DELATT/{0}/xmed".format(stacked_mesh.name))
        lg.sendline("cmo/DELATT/{0}/ymed".format(stacked_mesh.name))
        lg.sendline("cmo/DELATT/{0}/zmed".format(stacked_mesh.name))
        lg.sendline("cmo/DELATT/{0}/ikey".format(stacked_mesh.name))

        cmo_in = stacked_mesh.copy()

        # ==================================== #
        # Infiles._surf_mesh_backup(mesh_prism,mesh_surf,skip_sort=True)
        lg.sendline("extract/surfmesh/1,0,0/mo2/mo1/external")
        lg.sendline("dump/avs/surfmesh_lg.inp/mo2")
        # ==================================== #

        mo_surf = lg.read("surfmesh_lg.inp")
        cmo_in.delete()
        mo_surf.select()

        exported_fs = []

        # Generate basic top, side, and bottom sidesets
        naive = {"top": True, "bottom": True, "sides": True}
        if naive:
            new_fs = DEV__driver_naive(
                lg, mo_surf, naive["top"], naive["bottom"], naive["sides"]
            )

            exported_fs.extend(new_fs)

        # outfile = "EXODUSMESHOUT.exo"
        outfile = "GOLD.exo"
        cmd = f"dump/exo/{outfile}/{stacked_mesh.name}///facesets &\n"
        cmd += " &\n".join(exported_fs)
        lg.sendline(cmd)

        mesh_gold = "/Users/livingston/playground/tinerator/tinerator-core/example-data/meshes/Surfaces/simple_volume_with_facesets.exo"
        out1 = os.path.join(tmp_dir, "mesh_bronze.txt")
        out2 = os.path.join(tmp_dir, "mesh_gold.txt")
        os.system(f"ncdump {os.path.join(tmp_dir, outfile)} > {out1}")
        os.system(f"ncdump {mesh_gold} > {out2}")
        os.system(f"diff {out1} {out2}")


def extract_surface_mesh(mesh):  #: Mesh):
    """
    Extracts the boundary of a mesh. For a solid (volume) mesh,
    it extracts the surface mesh. If it is a surface mesh, it
    extracts the edge mesh.

    TODO: the current issue is that a surf mesh for a prism is
    multi-material (tri for surface; quad for sides.)
    """

    with tempfile.TemporaryDirectory() as tmp_dir:
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
