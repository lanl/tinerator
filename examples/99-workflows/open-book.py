import tinerator as tin
import numpy as np


def f(x, y, xv, zv, v_slope, y0, stream_slope):
    z = zv + v_slope * np.abs(x - xv) + (y - y0) * stream_slope
    return z


def make_mesh(
    stream_slope=0.05,
    nx=11,
    ny=21,
    x_min=0.0,
    x_max=10.0,
    y_min=0.0,
    y_max=20.0,
    z_min=0.0,
    z_v=7.0,
    z_max=10.0,
):
    x_v = (x_max - x_min) / 2.0
    dx = (x_max - x_min) / (nx - 1)
    dy = (y_max - y_min) / (ny - 1)

    v_slope = (z_max - z_v) / x_v

    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x, y)

    Z = f(X, Y, x_v, z_v, v_slope, y_min, stream_slope)

    return x, y, Z


x, y, Z = make_mesh(
    nx=12,
    ny=21,
    stream_slope=0.05,
    x_min=0.0,
    x_max=10.0,
    y_min=0.0,
    y_max=20.0,
    z_min=0.0,
    z_max=10.0,
    z_v=7.0,
)

quad_mesh = tin.meshing.create_hillslope_mesh(Z, x_coords=x, y_coords=y)
quad_mesh.plot()

z_min = 0.0
n_z = 11

layers = [
    ("snapped", z_min, n_z, 1),
]
hex_mesh = tin.meshing.extrude_mesh(quad_mesh, layers)
hex_mesh.plot()

surface_mesh = hex_mesh.surface_mesh()

set_top = surface_mesh.top_faces
sets_normals = surface_mesh.from_cell_normals()

sets = sets_normals + [set_top]

# Verify that all sets cover the whole of the mesh,
# and that there are no overlapping faces
test_sets = [surface_mesh.top_faces, surface_mesh.bottom_faces, surface_mesh.side_faces]
surface_mesh.validate_sets(test_sets)

hex_mesh.plot(sets=sets)
hex_mesh.save("Open-Book-3D.exo", sets=sets)
