import os
import numpy as np
import tempfile
from pylagrit import PyLaGriT
from copy import copy, deepcopy
from enum import Enum, auto
from .mesh import Mesh, StackedMesh, ElementType, load_mesh
from ..logging import log, warn, debug, _pylagrit_verbosity


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
