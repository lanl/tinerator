import os
import tinerator as tin
from helper import MESH_DIR


def test_simple():
    mesh = tin.meshing.load(os.path.join(MESH_DIR, "avs", "simple_surface.inp"))

    mesh_stacked = tin.meshing.layering.stack(
        mesh,
        tin.meshing.layering.uniform_sublayering(20.0, 1, matids=[1], relative_z=False),
    )
